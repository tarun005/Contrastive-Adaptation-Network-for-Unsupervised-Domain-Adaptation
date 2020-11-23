import torch
import torch.nn as nn
import os
from . import utils as solver_utils
from utils.utils import to_cuda, to_onehot
from torch import optim
from . import clustering
from discrepancy.cdd import CDD
from math import ceil as ceil
from .base_solver import BaseSolver
from copy import deepcopy
from . import criterion_factory as cf
from model import network
from model import lr_schedule


class CANSolver(BaseSolver):
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        super(CANSolver, self).__init__(net, dataloader, \
                      bn_domain_map=bn_domain_map, resume=resume, **kwargs)

        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 0}

        self.clustering_source_name = 'clustering_' + self.source_name
        self.clustering_target_name = 'clustering_' + self.target_name

        assert('categorical' in self.train_data)

        num_layers = len(self.net.module.FC) + 1
        self.cdd = CDD(kernel_num=self.opt.CDD.KERNEL_NUM, kernel_mul=self.opt.CDD.KERNEL_MUL,
                  num_layers=num_layers, num_classes=self.opt.DATASET.NUM_CLASSES, 
                  intra_only=self.opt.CDD.INTRA_ONLY)

        self.discrepancy_key = 'intra' if self.opt.CDD.INTRA_ONLY else 'cdd'
        self.clustering = clustering.Clustering(self.opt.CLUSTERING.EPS, 
                                        self.opt.CLUSTERING.FEAT_KEY, 
                                        self.opt.CLUSTERING.BUDGET)

        self.clustered_target_samples = {}
        sim_config = {
        'similarity_func' : 'cosine',
        'top_n_sim': 5,
        'ss_loss': False,
        'ranking_k': 3,
        'top_ranked_n': 10,
        'knn_method': 'ranking'
        }
        self.sim_module = cf.KnnSfmxConstLoss(sim_config)
        self.sim_module = self.sim_module.cuda()
        self.ILA_adver_create_network()



    def complete_training(self):
        if self.loop >= self.opt.TRAIN.MAX_LOOP:
            return True

        if 'target_centers' not in self.history or \
                'ts_center_dist' not in self.history or \
                'target_labels' not in self.history:
            return False

        if len(self.history['target_centers']) < 2 or \
		len(self.history['ts_center_dist']) < 1 or \
		len(self.history['target_labels']) < 2:
           return False

        # target centers along training
        target_centers = self.history['target_centers']
        eval1 = torch.mean(self.clustering.Dist.get_dist(target_centers[-1], 
			target_centers[-2])).item()

        # target-source center distances along training
        eval2 = self.history['ts_center_dist'][-1].item()

        # target labels along training
        path2label_hist = self.history['target_labels']
        paths = self.clustered_target_samples['data']
        num = 0
        for path in paths:
            pre_label = path2label_hist[-2][path]
            cur_label = path2label_hist[-1][path]
            if pre_label != cur_label:
                num += 1
        eval3 = 1.0 * num / len(paths)

        return (eval1 < self.opt.TRAIN.STOP_THRESHOLDS[0] and \
                eval2 < self.opt.TRAIN.STOP_THRESHOLDS[1] and \
                eval3 < self.opt.TRAIN.STOP_THRESHOLDS[2])

#     def ILA_solve(self):
#         stop = False
#         if self.resume:
#             self.iters += 1
#             self.loop += 1

#         while True: 
#             # updating the target label hypothesis through clustering
#             target_hypt = {}
#             filtered_classes = []
#             with torch.no_grad():
#                 take model 
        
        
    def solve(self):
        stop = False
        if self.resume:
            self.iters += 1
            self.loop += 1

        while True: 
            # updating the target label hypothesis through clustering
            target_hypt = {}
            filtered_classes = []
            with torch.no_grad():
                #self.update_ss_alignment_loss_weight()
                print('Clustering based on %s...' % self.source_name)
                self.update_labels()
                self.clustered_target_samples = self.clustering.samples
                target_centers = self.clustering.centers 
                center_change = self.clustering.center_change 
                path2label = self.clustering.path2label

                # updating the history
                self.register_history('target_centers', target_centers,
	            	self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('ts_center_dist', center_change,
	            	self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('target_labels', path2label,
	            	self.opt.CLUSTERING.HISTORY_LEN)

                if self.clustered_target_samples is not None and \
                              self.clustered_target_samples['gt'] is not None:
                    preds = to_onehot(self.clustered_target_samples['label'], 
                                                self.opt.DATASET.NUM_CLASSES)
                    gts = self.clustered_target_samples['gt']
                    res = self.model_eval(preds, gts)
                    print('Clustering %s: %.4f' % (self.opt.EVAL_METRIC, res))

                # check if meet the stop condition
                stop = self.complete_training()
                if stop: break
                
                # filtering the clustering results
                target_hypt, filtered_classes = self.filtering()

                # update dataloaders
                self.construct_categorical_dataloader(target_hypt, filtered_classes)
                # update train data setting
                self.compute_iters_per_loop(filtered_classes)

            # k-step update of network parameters through forward-backward process
            # self.ILA_update_network(filtered_classes)
            self.ILA_adver_update_network(filtered_classes)
            self.loop += 1

        print('Training Done!')
        
    
    def update_labels(self):
        net = self.net
        net.eval()
        opt = self.opt

        source_dataloader = self.train_data[self.clustering_source_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.source_name])

        source_centers = solver_utils.get_centers(net, 
		source_dataloader, self.opt.DATASET.NUM_CLASSES, 
                self.opt.CLUSTERING.FEAT_KEY)
        init_target_centers = source_centers

        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.target_name])

        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(net, target_dataloader)

    def filtering(self):
        threshold = self.opt.CLUSTERING.FILTERING_THRESHOLD
        min_sn_cls = self.opt.TRAIN.MIN_SN_PER_CLASS
        target_samples = self.clustered_target_samples

        # filtering the samples
        chosen_samples = solver_utils.filter_samples(target_samples, threshold=threshold)

        # filtering the classes
        filtered_classes = solver_utils.filter_class(
		chosen_samples['label'], min_sn_cls, self.opt.DATASET.NUM_CLASSES)

        print('The number of filtered classes: %d.' % len(filtered_classes))
        return chosen_samples, filtered_classes
    
#     def ILA_labelling_and_filtering(self):
# #         take network, do KNN labelling and filtering
#         labels_tgt = cf.sort_div(out_src, out_tgt, src_labels)
#         filtered_labels_tgts = cf.filtering(sim_matrix, src_labels, labels_tgt)
#         filtered_samples['data'] = filtered_data #tgt features
#         filtered_samples['label'] = filtered_label #predicted label
#         filtered_samples['gt'] = filtered_gt #ground truth
#         self.filter_class
        
    
    def construct_categorical_dataloader(self, samples, filtered_classes):
        # update self.dataloader
        target_classwise = solver_utils.split_samples_classwise(
			samples, self.opt.DATASET.NUM_CLASSES)

        dataloader = self.train_data['categorical']['loader']
        classnames = dataloader.classnames
        dataloader.class_set = [classnames[c] for c in filtered_classes]
        dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] \
                      for c in filtered_classes}
        dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.construct()

    def CAS(self):
        samples = self.get_samples('categorical')

        source_samples = samples['Img_source']
        source_sample_paths = samples['Path_source']
        source_nums = [len(paths) for paths in source_sample_paths]

        target_samples = samples['Img_target']
        target_sample_paths = samples['Path_target']
        target_nums = [len(paths) for paths in target_sample_paths]
        
        source_sample_labels = samples['Label_source']
        self.selected_classes = [labels[0].item() for labels in source_sample_labels]
        assert(self.selected_classes == 
               [labels[0].item() for labels in  samples['Label_target']])
        return source_samples, source_nums, target_samples, target_nums

    def ILA_CAS(self):
        samples = self.get_samples('categorical')

        source_samples = samples['Img_source']
        source_sample_paths = samples['Path_source']
        source_nums = [len(paths) for paths in source_sample_paths]
#         print('samples={}'.format(samples))
#         print('source_samples shape={}'.format(source_samples.shape))


        target_samples = samples['Img_target']
        target_sample_paths = samples['Path_target']
        target_nums = [len(paths) for paths in target_sample_paths]
#         print('target_samples shape={}'.format(target_samples.shape))

        source_sample_labels = samples['Label_source']
        self.selected_classes = [labels[0].item() for labels in source_sample_labels]

        src_labels = torch.cat(source_sample_labels, dim=0)
        src_labels = src_labels.cuda()
        
        tgt_labels_pred = torch.cat(samples['Label_target'], dim=0)
        tgt_labels_pred = tgt_labels_pred.cuda()
        assert (self.selected_classes == [labels[0].item() for labels in samples['Label_target']])
        
#         print('samples[Label_source]={}, samples[Label_target]={}, src_labels={}, tgt_labels_pred={}'.format(\
#                 samples['Label_source'], samples['Label_target'], src_labels, tgt_labels_pred))
        return source_samples, source_nums, target_samples, target_nums, src_labels, tgt_labels_pred

    def prepare_feats(self, feats):
        return [feats[key] for key in feats if key in self.opt.CDD.ALIGNMENT_FEAT_KEYS]

    def compute_iters_per_loop(self, filtered_classes):
        self.iters_per_loop = int(len(self.train_data['categorical']['loader'])) * self.opt.TRAIN.UPDATE_EPOCH_PERCENTAGE
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def update_network(self, filtered_classes):
        # initial configuration
        stop = False
        update_iters = 0

        self.train_data[self.source_name]['iterator'] = \
                     iter(self.train_data[self.source_name]['loader'])
        self.train_data['categorical']['iterator'] = \
                     iter(self.train_data['categorical']['loader'])

        while not stop:
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()
            self.net.zero_grad()

            loss = 0
            ce_loss_iter = 0
            cdd_loss_iter = 0

            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.source_name) 
            source_data, source_gt = source_sample['Img'],\
                          source_sample['Label']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_preds = self.net(source_data)['logits']

            # compute the cross-entropy loss
            ce_loss = self.CELoss(source_preds, source_gt)
            ce_loss.backward()

            ce_loss_iter += ce_loss
            loss += ce_loss
         
            if len(filtered_classes) > 0:
                # update the network parameters
                # 1) class-aware sampling
                source_samples_cls, source_nums_cls, \
                       target_samples_cls, target_nums_cls = self.CAS()

                # 2) forward and compute the loss
                source_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in source_samples_cls], dim=0)
                target_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in target_samples_cls], dim=0)

                self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
                feats_source = self.net(source_cls_concat)
                self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                feats_target = self.net(target_cls_concat)

                # prepare the features
                feats_toalign_S = self.prepare_feats(feats_source)
                feats_toalign_T = self.prepare_feats(feats_target)                 

                cdd_loss = self.cdd.forward(feats_toalign_S, feats_toalign_T, 
                               source_nums_cls, target_nums_cls)[self.discrepancy_key]

                cdd_loss *= self.opt.CDD.LOSS_WEIGHT
                cdd_loss.backward()

                cdd_loss_iter += cdd_loss
                loss += cdd_loss

            # update the network
            self.optimizer.step()

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                # accu = self.model_eval(source_preds, source_gt)
                cur_loss = {'ce_loss': ce_loss_iter, 'cdd_loss': cdd_loss_iter,
			'total_loss': loss}
                self.logging(cur_loss, 0)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    accu = self.test()
                    print('Test at (loop %d, iters: %d) with %s: %.4f.' % (self.loop, 
                              self.iters, self.opt.EVAL_METRIC, accu))

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False

    def ILA_update_network(self, filtered_classes):
        # initial configuration
        print("ILA update network at loop {}".format(self.loop))
        stop = False
        update_iters = 0

        self.train_data[self.source_name]['iterator'] = \
            iter(self.train_data[self.source_name]['loader'])
        self.train_data['categorical']['iterator'] = \
            iter(self.train_data['categorical']['loader'])
        all_total = 0.0
        all_correct = 0.0
        filtered_total = 0.0
        filtered_correct = 0.0
    
        while not stop:
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()
            self.net.zero_grad()

            loss = 0
            ce_loss_iter = 0
            cdd_loss_iter = 0

            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.source_name)
            source_data, source_gt = source_sample['Img'], \
                                     source_sample['Label']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_preds = self.net(source_data)['logits']

            # compute the cross-entropy loss
            ce_loss = self.CELoss(source_preds, source_gt)
            ce_loss.backward()

            ce_loss_iter += ce_loss
            loss += ce_loss

            if len(filtered_classes) > 0:
                # update the network parameters
                # 1) class-aware sampling
                source_samples_cls, source_nums_cls, \
                target_samples_cls, target_nums_cls, src_labels, tgt_labels_pred = self.ILA_CAS()

                # 2) forward and compute the loss
                source_cls_concat = torch.cat([to_cuda(samples)
                                               for samples in source_samples_cls], dim=0)
                target_cls_concat = torch.cat([to_cuda(samples)
                                               for samples in target_samples_cls], dim=0)

#                 print('type source_samples_cls={}, type target_samples_cls={}'.format(type(source_samples_cls), \
#                                                                                       type(target_samples_cls)))
#                 print("source_cls_concat={}, target_cls_concat={}".format(source_cls_concat, target_cls_concat))
                self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
                feats_source = self.net(source_cls_concat)
                self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
#                 print('source in shape'.format(source_cls_concat.shape))
                feats_target = self.net(target_cls_concat)
#                 print('type feats_source = {}, type feats_target'.format(type(feats_source), type(feats_target)))
                # prepare the features
#                 print('shape in target'.format(target_cls_concat.shape))
                feats_toalign_S = self.prepare_feats(feats_source)
                feats_toalign_T = self.prepare_feats(feats_target)

                if self.loop >= self.opt.TRAIN.LOOPS_BEFORE_ILA:
                    fs, ft = feats_toalign_S[0], feats_toalign_T[0]
#                     for fs, ft in zip(feats_toalign_S, feats_toalign_T):
#                         print("fs shape={}, ft shape={}".format(fs.shape, ft.shape))
                    assert fs.shape[0] == ft.shape[0]
#                         assert fs.shape[0] == 30, 'num of features source={}'.format(fs.shape[0])
#                         assert src_labels.shape[0] == 30, 'num of labels source={}'.format(src_labels.shape[0])
                    f_st = torch.cat((fs, ft), dim=0)
                    print('f_st shape={}'.format(f_st.shape))
                    sim_loss = 2.0* (self.sim_module(f_st, criterion_inputs= {'src_labels': src_labels}))
#                         sim_matrix = self.sim_module.get_sim_matrix(fs, ft)
#                         sim_matrix = sim_matrix.cuda()
#                         sim_loss = self.sim_module.calc_loss_rect_matrix(sim_matrix, src_labels, tgt_labels_pred)
                    conf_ind = self.sim_module.conf_ind
                    all_assigned = self.sim_module.all_assigned
                    all_total += all_assigned.shape[0]
                    all_correct += torch.sum((all_assigned == tgt_labels_pred).float()).item()
                    fil_labels = tgt_labels_pred[conf_ind] 
                    fil_assigned = all_assigned[conf_ind]
                    filtered_correct += torch.sum((fil_labels == fil_assigned).float()).item()
                    filtered_total += conf_ind.shape[0]
                    del all_assigned, fil_labels, fil_assigned
#                     print('fil corr: {}, fil total: {}, all_correct:{}, all_total: {}'.format(filtered_correct, filtered_total, all_correct, all_total ) )
#                     print('conf_ind = {}, all_assigned = {}'.format(conf_ind, all_assigned))

                    assert (torch.isnan(sim_loss) == False)
                    sim_loss.backward(retain_graph=True)
                    

                cdd_loss = self.cdd.forward(feats_toalign_S, feats_toalign_T,
                                            source_nums_cls, target_nums_cls)[self.discrepancy_key]

                cdd_loss *= self.opt.CDD.LOSS_WEIGHT
                cdd_loss.backward()

                cdd_loss_iter += cdd_loss
                loss += cdd_loss

            # update the network
            self.optimizer.step()
            del cdd_loss
            if self.loop >= self.opt.TRAIN.LOOPS_BEFORE_ILA:
                del sim_loss

            if self.opt.TRAIN.LOGGING and (update_iters + 1) % \
                    (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                # accu = self.model_eval(source_preds, source_gt)
                cur_loss = {'ce_loss'   : ce_loss_iter, 'cdd_loss': cdd_loss_iter,
                            'total_loss': loss}
                self.logging(cur_loss, 0)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
                    (update_iters + 1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    accu = self.test()
                    print('Test at (loop %d, iters: %d) with %s: %.4f.' % (self.loop,
                                                                           self.iters, self.opt.EVAL_METRIC, accu))
                    if self.loop >= self.opt.TRAIN.LOOPS_BEFORE_ILA:
                        print("loop: {:05d}, all_precision: {:.5f}, filtered_precision: {:.5f},".format(self.loop, 100.0*all_correct/all_total, 100.0*filtered_correct/filtered_total ))
                        all_total = 0.0
                        all_correct = 0.0
                        filtered_total = 0.0
                        filtered_correct = 0.0

#             if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
#                     (update_iters + 1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
#                 self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False

    # def stage(self):
    #     if config["loss"]["random"]:
    #         random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
    #         ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    #     else:
    #         random_layer = None
    #         if config['method'] == 'DANN':
    #             ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)
    #         else:
    #             ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    #     if config["loss"]["random"]:
    #         random_layer.cuda()
    #     ad_net = ad_net.cuda()
    #     parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    #     classifier_param_list = base_network.get_parameters()
    #
    #     ad_net.train(True)
    #     optimizer = lr_scheduler(optimizer, j, **schedule_param)
    #
    #     features = torch.cat((all_src_features, all_tgt_features), dim=0)
    #     outputs = torch.cat((all_out_src, all_out_tgt), dim=0)
    #     softmax_out = nn.Softmax(dim=1)(outputs)
    #
    #     if config['method'] == 'CDAN+E':
    #         print('using method CDAN+E')
    #         entropy = loss.Entropy(softmax_out)
    #         transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
    #     elif config['method'] == 'CDAN':
    #         print('using method CDAN')
    #         transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
    #     elif config['method'] == 'DANN':
    #         print('using method DANN')
    #         transfer_loss = loss.DANN(features, ad_net)
    #     else:
    #         raise ValueError('Method cannot be recognized.')

    def ILA_adver_create_network(self):
        # self.method
        # import network
        if self.method == 'DANN':
            self.ad_net = network.AdversarialNetwork(2048, 1024)
        elif self.method == 'CDAN':
            self.ad_net = network.AdversarialNetwork(2048 * 31, 1024)
        lr = 0.001
        optimizer_config = {"type": optim.SGD, "optim_params": {'lr'          : lr, "momentum": 0.9, \
                                                                   "weight_decay": 0.0005, "nesterov": True},
                               "lr_type": "inv", \
                               "lr_param": {"lr": lr, "gamma": 0.001, "power": 0.75}}

        parameter_list = self.ad_net.get_parameters()
        self.optimizer_adv = optimizer_config["type"](parameter_list, \
                                             **(optimizer_config["optim_params"]))
        param_lr = []
        for param_group in self.optimizer_adv.param_groups:
            param_lr.append(param_group["lr"])
        self.schedule_param = optimizer_config["lr_param"]
        self.lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
        self.ad_net.train(True)

        self.ad_net = to_cuda(self.ad_net)

        self.ad_net.train(True)


    def ILA_adver_update_network(self, filtered_classes):
        # initial configuration
        print("ILA update network at loop {}".format(self.loop))
        stop = False
        update_iters = 0

        self.train_data[self.source_name]['iterator'] = \
            iter(self.train_data[self.source_name]['loader'])

        self.train_data[self.target_name]['iterator'] = \
            iter(self.train_data[self.target_name]['loader'])

        self.train_data['categorical']['iterator'] = \
            iter(self.train_data['categorical']['loader'])
        all_total = 0.0
        all_correct = 0.0
        filtered_total = 0.0
        filtered_correct = 0.0

        while not stop:
            # update learning rate
            self.update_lr()
            self.optimizer_adv = self.lr_scheduler(self.optimizer_adv, self.loop, **self.schedule_param)
            self.optimizer_adv.zero_grad()

            # set the status of network
            self.net.train()
            self.net.zero_grad()

            loss = 0
            ce_loss_iter = 0
            cdd_loss_iter = 0

            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.source_name)
            source_data, source_gt = source_sample['Img'], \
                                     source_sample['Label']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            feats_source = self.net(source_data)

            # compute the cross-entropy loss
            ce_loss = total_loss = self.CELoss(feats_source['logits'], source_gt)
            # ce_loss.backward()

            ce_loss_iter += ce_loss
            loss += ce_loss


            target_sample = self.get_samples(self.target_name)
            target_data, target_gt = target_sample['Img'], \
                                     target_sample['Label']
            target_data = to_cuda(target_data)
            target_gt = to_cuda(target_gt)

            # self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            # feats_source = self.net(source_data)
            self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
            #                 print('source in shape'.format(source_cls_concat.shape))
            feats_target = self.net(target_data)

            feats_toalign_S = self.prepare_feats(feats_source)
            feats_toalign_T = self.prepare_feats(feats_target)

            features = torch.cat((feats_toalign_S[0], feats_toalign_T[0]), dim=0)
            outputs = torch.cat((feats_toalign_S[1], feats_toalign_T[1]), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

            if self.method == 'CDAN':
                print('using method CDAN')
                transfer_loss = loss.CDAN([features, softmax_out], self.ad_net, None, None, None)
            elif self.method == 'DANN':
                print('using method DANN')
                transfer_loss = loss.DANN(features, self.ad_net)
            else:
                raise ValueError('Method cannot be recognized.')

            total_loss += transfer_loss

            if len(filtered_classes) > 0:
                # update the network parameters
                # 1) class-aware sampling
                source_samples_cls, source_nums_cls, \
                target_samples_cls, target_nums_cls, src_labels, tgt_labels_pred = self.ILA_CAS()

                # 2) forward and compute the loss
                source_cls_concat = torch.cat([to_cuda(samples)
                                               for samples in source_samples_cls], dim=0)
                target_cls_concat = torch.cat([to_cuda(samples)
                                               for samples in target_samples_cls], dim=0)

                #                 print('type source_samples_cls={}, type target_samples_cls={}'.format(type(source_samples_cls), \
                #                                                                                       type(target_samples_cls)))
                #                 print("source_cls_concat={}, target_cls_concat={}".format(source_cls_concat, target_cls_concat))
                self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
                feats_source = self.net(source_cls_concat)
                self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                #                 print('source in shape'.format(source_cls_concat.shape))
                feats_target = self.net(target_cls_concat)
                #                 print('type feats_source = {}, type feats_target'.format(type(feats_source), type(feats_target)))
                # prepare the features
                #                 print('shape in target'.format(target_cls_concat.shape))
                feats_toalign_S = self.prepare_feats(feats_source)
                feats_toalign_T = self.prepare_feats(feats_target)

                if self.loop >= self.opt.TRAIN.LOOPS_BEFORE_ILA:
                    fs, ft = feats_toalign_S[0], feats_toalign_T[0]
                    #                     for fs, ft in zip(feats_toalign_S, feats_toalign_T):
                    #                         print("fs shape={}, ft shape={}".format(fs.shape, ft.shape))
                    assert fs.shape[0] == ft.shape[0]
                    #                         assert fs.shape[0] == 30, 'num of features source={}'.format(fs.shape[0])
                    #                         assert src_labels.shape[0] == 30, 'num of labels source={}'.format(src_labels.shape[0])
                    f_st = torch.cat((fs, ft), dim=0)
                    print('f_st shape={}'.format(f_st.shape))
                    sim_matrix = self.sim_module.get_sim_matrix(fs, ft)
                    sim_matrix = sim_matrix.cuda()
                    sim_loss = 2.0 * (self.sim_module.calc_loss_rect_matrix(sim_matrix, src_labels, tgt_labels_pred))
                    # sim_loss = 2.0 * (self.sim_module(f_st, criterion_inputs={'src_labels': src_labels}))
                    #                         sim_matrix = self.sim_module.get_sim_matrix(fs, ft)
                    #                         sim_matrix = sim_matrix.cuda()
                    #                         sim_loss = self.sim_module.calc_loss_rect_matrix(sim_matrix, src_labels, tgt_labels_pred)
                    conf_ind = self.sim_module.conf_ind
                    all_assigned = self.sim_module.all_assigned
                    all_total += all_assigned.shape[0]
                    all_correct += torch.sum((all_assigned == tgt_labels_pred).float()).item()
                    fil_labels = tgt_labels_pred[conf_ind]
                    fil_assigned = all_assigned[conf_ind]
                    filtered_correct += torch.sum((fil_labels == fil_assigned).float()).item()
                    filtered_total += conf_ind.shape[0]
                    del all_assigned, fil_labels, fil_assigned
                    #                     print('fil corr: {}, fil total: {}, all_correct:{}, all_total: {}'.format(filtered_correct, filtered_total, all_correct, all_total ) )
                    #                     print('conf_ind = {}, all_assigned = {}'.format(conf_ind, all_assigned))

                    assert (torch.isnan(sim_loss) == False)
                    # sim_loss.backward(retain_graph=True)
                    total_loss += sim_loss
                    # sim_loss.backward()

                # cdd_loss = self.cdd.forward(feats_toalign_S, feats_toalign_T,
                #                             source_nums_cls, target_nums_cls)[self.discrepancy_key]
                #
                # cdd_loss *= self.opt.CDD.LOSS_WEIGHT
                # cdd_loss.backward()
                #
                # cdd_loss_iter += cdd_loss
                # loss += cdd_loss

            # update the network
            total_loss.backward()
            self.optimizer.step()
            self.optimizer_adv.step()
            # del cdd_loss
            if self.loop >= self.opt.TRAIN.LOOPS_BEFORE_ILA:
                del sim_loss

            if self.opt.TRAIN.LOGGING and (update_iters + 1) % \
                    (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                # accu = self.model_eval(source_preds, source_gt)
                cur_loss = {'ce_loss'   : ce_loss_iter, 'cdd_loss': cdd_loss_iter,
                            'total_loss': loss}
                self.logging(cur_loss, 0)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
                    (update_iters + 1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    accu = self.test()
                    print('Test at (loop %d, iters: %d) with %s: %.4f.' % (self.loop,
                                                                           self.iters, self.opt.EVAL_METRIC, accu))
                    if self.loop >= self.opt.TRAIN.LOOPS_BEFORE_ILA:
                        print("loop: {:05d}, all_precision: {:.5f}, filtered_precision: {:.5f},".format(self.loop,
                                                                                                        100.0 * all_correct / all_total,
                                                                                                        100.0 * filtered_correct / filtered_total))
                        all_total = 0.0
                        all_correct = 0.0
                        filtered_total = 0.0
                        filtered_correct = 0.0

            #             if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
            #                     (update_iters + 1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
            #                 self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False