import torch.nn as nn
import torch.nn.functional as F
import torch
import math
# from pytorch_metric_learning import losses, miners, distances, reducers, testers
# from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class KnnSfmxConstLoss(nn.Module):
    def __init__(self, config_data):
        super(KnnSfmxConstLoss, self).__init__()
        self.ranking_method = 'sim_ratio'
        self.ranking_k = config_data['ranking_k']
        self.top_ranked_n = config_data['top_ranked_n']
        # Needed
        #self.src_n_cls = config_data['dataset']['num_classes']
        #self.emb_size = config_data['model']["similarity_net"]["out_emb_size"]
        self.eps = 1e-9
        self.similarity_func = config_data['similarity_func']  # euclidean dist, cosine
        #self.margin = config_data['criterion']['margin']
        self.gen_assigned_tgt_labels = None
#         if self.similarity_func == 'euclidean':
#             direct_proportional = False
#         elif self.similarity_func == 'cosine':
#             direct_proportional = True
#         self.confidence_margins = {'direct_proportional': direct_proportional}
        self.top_n_sim = config_data['top_n_sim']
        self.ss_loss = config_data['ss_loss']
        self.knn_method = config_data['knn_method']
        
        self.all_assigned = None
        self.conf_ind = None
        
        

    #     @property
    #     def gen_assigned_tgt_labels(self):
    #         return self.__assigned_tar_labels

    def get_sim_matrix(self, out_src, out_tar):
        # Source X Target similarity
        matrix = None
        if (self.similarity_func == 'euclidean'):
            matrix = torch.cdist(out_src, out_tar)
            matrix = matrix + 1.0
            matrix = 1.0/matrix
            assert torch.min(matrix).item() >= 0.0, 'similarity matrix has negative values, euclidean'

        if (self.similarity_func == 'cosine'):
            # matrix = torch.zeros((n, n), dtype=torch.float64)
            out_src = F.normalize(out_src, dim=1, p=2)
            out_tar = F.normalize(out_tar, dim=1, p=2)
            matrix = torch.matmul(out_src, out_tar.T)
            matrix = 0.5*(matrix + 1.0)
            assert torch.min(matrix).item() >= 0.0, 'similarity matrix has negative values, cosine'
            # print('matrix {}'.format(matrix))
            # source X target matrix as in overleaf figure
        return matrix

    def __target_labels_sort_div(self, sim_matrix, src_labels):
#         sim_matrix = dist_matrix if (self.confidence_margins['direct_proportional']) else -1. * dist_matrix
        ind = torch.sort(sim_matrix, descending=True, dim=0).indices
        ind_split = torch.split(ind, 1, dim=1)
        ind_split = [id.squeeze() for id in ind_split]
        vr_src = src_labels.unsqueeze(-1).repeat(1, self.n_per_domain)
        label_list = []
        for i in range(0, self.n_per_domain):
            _row = ind_split[i].long()
            _col = (torch.ones(self.n_per_domain) * i).long()
            _val = vr_src[_row, _col]
            top_n_val = _val[[j for j in range(0, self.top_n_sim)]]
            label_list.append(top_n_val)

        all_top_labels = torch.stack(label_list, dim=1)
        assigned_target_labels = torch.mode(all_top_labels, dim=0).values
        return assigned_target_labels
    
    
    def __calc_ss_loss(self, out_src, src_labels):
        n = self.n_per_domain
        vr_src = src_labels.unsqueeze(-1).repeat(1, n)
        hr_src = src_labels.unsqueeze(-2).repeat(n, 1)
        label_ss = (vr_src == hr_src)
        idx = torch.arange(0, n, out=torch.LongTensor())

        label_ss[idx, idx] = False
        sim_labels_fl = label_ss.float()
        _sum = torch.sum(sim_labels_fl, dim=1)
        _sum = _sum > 0.
        mask_ghost = [torch.ones(n) if (_s) else torch.zeros(n) for _s in _sum]
        mask_ghost = torch.stack(mask_ghost, dim=0)

        if torch.cuda.is_available():
            mask_ghost = mask_ghost.cuda()

        final_mask = mask_ghost.bool()
        final_mask[idx, idx] = False

        similarity_func = self.similarity_func
        if (similarity_func == 'euclidean'):
            dist_src = torch.cdist(out_src, out_src)
            dist_src = dist_src + 1.0
            sim_matrix = 1.0 /dist_src

        if (similarity_func == 'cosine'):
            # matrix = torch.zeros((n, n), dtype=torch.float64)
            out_src_norm = F.normalize(out_src, dim=1, p=2)
            dist_src = torch.matmul(out_src_norm, out_src_norm.T)
            sim_matrix = dist_src + 1.0

#         sim_matrix = dist_src if (self.confidence_margins['direct_proportional']) else -1. * dist_src
        sim_matrix[~final_mask] = float('-inf')
        sft_matrix = torch.softmax(sim_matrix, dim=1)

        filtered_sft_matrix = sft_matrix[~torch.isnan(sft_matrix.sum(dim=1))]
        filtered_sim_labels = sim_labels_fl[~torch.isnan(sft_matrix.sum(dim=1))]

        if (filtered_sft_matrix.shape[0] == 0.):
            loss = torch.tensor(0., requires_grad=True).float()
            if torch.cuda.is_available():
                loss = loss.cuda()
            return loss

        num = torch.sum(filtered_sft_matrix * filtered_sim_labels, dim=1)
        den = torch.sum(filtered_sft_matrix, dim=1)
        # FILTER OUT when all are positives for a source
        mean_loss = -1 * torch.mean(torch.log(num / den))
        return mean_loss
    
    
    def forward_ranking(self, output, size_average=True, margin_update=0.,
                criterion_inputs=None, shift=0):
#         out_src, out_tar = output
#         ns = 
        n = output.shape[0] / 2  # n number of samples from source and n from target (batch_size = 2*n)
        n = int(n)
        self.n_per_domain = n
        out_src, out_tar = torch.split(output, int(n), dim=0)
#         assert n==30

        sim_matrix = self.get_sim_matrix(out_src, out_tar)
        src_labels = criterion_inputs['src_labels']
        assigned_tgt_labels = self.__target_labels_sort_div(sim_matrix, src_labels)
        self.gen_assigned_tgt_labels = assigned_tgt_labels
        self.all_assigned = assigned_tgt_labels
        #select confident target samples
#         sim_matrix = dist_matrix if (self.confidence_margins['direct_proportional']) else -1. * dist_matrix
#         sim_sorted, ind = torch.sort(sim_matrix, descending=True, dim=0)
        
#         ind_split = torch.split(ind, 1, dim=1)
#         ind_split = [id.squeeze() for id in ind_split]
        
#         sim_sorted_split = torch.split(sim_sorted, 1, dim=1)
#         sim_sorted_split = [id.squeeze() for id in sim_sorted_split]
        
#         vr_src = src_labels.unsqueeze(-1).repeat(1, self.n_per_domain)
        ranking_score_list = []
        flat_src_labels = src_labels.squeeze()
        
        sim_matrix_split = torch.split(sim_matrix, 1, dim=1)
        sim_matrix_split = [_id.squeeze() for _id in sim_matrix_split]
        r = self.ranking_k
        for i in range(0, n):
            t_label = assigned_tgt_labels[i]
            nln_mask = (flat_src_labels == t_label)
            nln_sim_all = sim_matrix_split[i][nln_mask]
            nln_sim_r = torch.narrow(torch.sort(nln_sim_all, descending=True)[0], 0, 0, r)
            
            nun_mask = ~(flat_src_labels == t_label)
            nun_sim_all = sim_matrix_split[i][nun_mask]
            nun_sim_r = torch.narrow(torch.sort(nun_sim_all, descending=True)[0], 0, 0, r)
            
            pred_conf_score = (1.0*torch.sum(nln_sim_r)/torch.sum(nun_sim_r)).item()
            ranking_score_list.append(pred_conf_score)
        
        sort_ranking_score, ind_tgt = torch.sort(torch.tensor(ranking_score_list), descending=True)
        print('sort_ranking_score={}, ind_tgt={}'.format(sort_ranking_score, ind_tgt))
        top_n_tgt_ind = torch.narrow(ind_tgt, 0, 0+shift, self.top_ranked_n)
        
        confident_sim_matrix_list = []
        
        for idx in top_n_tgt_ind:
            confident_sim_matrix_list.append(sim_matrix_split[idx])
            
        confident_sim_matrix = torch.stack(confident_sim_matrix_list, dim=1)
        confident_tgt_labels = assigned_tgt_labels[top_n_tgt_ind]        
#         print('confident_sim_matrix = {}, confident_tgt_labels= {}'.format(confident_sim_matrix, confident_tgt_labels))
        self.conf_ind = top_n_tgt_ind
        loss = self.calc_loss_rect_matrix(confident_sim_matrix, src_labels, confident_tgt_labels)
        return loss
        
        #calculate loss
    def calc_loss_rect_matrix(self, confident_sim_matrix, src_labels, confident_tgt_labels):
        
        n_src = src_labels.shape[0]
        n_tgt = confident_tgt_labels.shape[0]
        
        print("n_src={}, n_tgt={}".format(n_src, n_tgt))
        
        vr_src = src_labels.unsqueeze(-1).repeat(1, n_tgt)
        hr_tgt = confident_tgt_labels.unsqueeze(-2).repeat(n_src, 1)
        
        mask_sim = (vr_src == hr_tgt).float()
        mask_dis = (~(vr_src == hr_tgt)).float()
        
        sim_sum = torch.sum(mask_sim, dim=1)
        sim_sum = sim_sum > 0.
        
        dis_sum = torch.sum(mask_dis, dim=1)
        dis_sum = dis_sum > 0.
        
        ghost_sim = [torch.ones(n_tgt) if (_s) else torch.zeros(n_tgt) for _s in sim_sum]
        ghost_dis = [torch.ones(n_tgt) if (_s) else torch.zeros(n_tgt) for _s in dis_sum]
        
        ghost_sim = torch.stack(ghost_sim, dim=0)
        ghost_dis = torch.stack(ghost_dis, dim=0)
        
        if torch.cuda.is_available():
            ghost_sim, ghost_dis, mask_sim = ghost_sim.cuda(), ghost_dis.cuda(), mask_sim.cuda()
        
        final_mask = (ghost_sim.bool() & ghost_dis.bool()) #& mask_sim.bool()
#         print("final_mask={}".format(final_mask))
        if torch.cuda.is_available():
            confident_sim_matrix, final_mask = confident_sim_matrix.cuda(), final_mask.cuda()
        
        confident_sim_matrix[~final_mask] = float('-inf')
        sft_matrix = torch.softmax(confident_sim_matrix, dim=1)
        
        filtered_sft_matrix = sft_matrix[~torch.isnan(sft_matrix.sum(dim=1))]
        filtered_sim_labels = mask_sim[~torch.isnan(sft_matrix.sum(dim=1))]
           
#         print('filtered sft matrix {}'.format(filtered_sft_matrix))
#         print('filtered_sim_labels = {}'.format(filtered_sim_labels))
        
        num = torch.sum(filtered_sft_matrix * filtered_sim_labels, dim=1)
        den = torch.sum(filtered_sft_matrix, dim=1)
        
#         print('num= {}'.format(num))
#         print('den = {}'.format(den))

        mean_loss = -1 * torch.mean(torch.log(num / den))
        if self.ss_loss:
            raise Exception('SS loss not implemented with KNN confidence filtering')
        return mean_loss
        
           
    def forward(self, output, size_average=True, margin_update=0.,
                criterion_inputs=None, shift= 0):
        if self.knn_method == 'classic':
            return self.forward_classic(output, size_average=size_average, margin_update=margin_update,
                criterion_inputs=criterion_inputs, shift=shift)
        elif self.knn_method == 'ranking':
            return self.forward_ranking(output, size_average=size_average, margin_update=margin_update,
                criterion_inputs=criterion_inputs, shift=shift)
             
    
    # based on __mean_dist_analysis
    def forward_classic(self, output, size_average=True, margin_update=0.,
                criterion_inputs=None, shift = 0):  # criterion inputs ONLY SRC LA!!!!!
        # calculate the batch_src_mean_emb
        n = output.shape[0] / 2  # n number of samples from source and n from target (batch_size = 2*n)
        n = int(n)
        self.n_per_domain = n
        out_src, out_tar = torch.split(output, int(n), dim=0)

        sim_matrix = self.get_sim_matrix(out_src, out_tar)
        src_labels = criterion_inputs['src_labels']
        assigned_tgt_labels = self.__target_labels_sort_div(sim_matrix, src_labels)
        self.gen_assigned_tgt_labels = assigned_tgt_labels
        #         self.__assigned_tar_labels = assigned_tgt_labels
        vr_src = src_labels.unsqueeze(-1).repeat(1, n)
        hr_tgt = assigned_tgt_labels.unsqueeze(-2).repeat(n, 1)
        label_st = (vr_src == hr_tgt)  # getting positive True, negative pair False

#         sim_matrix = dist_matrix if (self.confidence_margins['direct_proportional']) else -1. * dist_matrix

        fl_sim_labels = label_st.float()
        _sum = torch.sum(fl_sim_labels, dim=1)
        _sum = _sum > 0.
        mask_ghost = [torch.ones(n) if (_s) else torch.zeros(n) for _s in _sum]
        mask_ghost = torch.stack(mask_ghost, dim=0)

        num_mask_ghost = (~_sum).float().sum().item()

        if torch.cuda.is_available():
            mask_ghost = mask_ghost.cuda()

        final_mask = mask_ghost.bool()
        # print('final mask: {}'.format(final_mask))

        if torch.cuda.is_available():
            sim_matrix, fl_sim_labels, final_mask = sim_matrix.cuda(), fl_sim_labels.cuda(), final_mask.cuda()

        # print('shape of matrix {}'.format(sim_matrix.shape))
        sim_matrix[~final_mask] = float('-inf')
        # print('matrix after inf {}'.format(sim_matrix))
        sft_matrix = torch.softmax(sim_matrix, dim=1)
        # print('sft matrix {}'.format(sft_matrix))
        filtered_sft_matrix = sft_matrix[~torch.isnan(sft_matrix.sum(dim=1))]
        filtered_sim_labels = fl_sim_labels[~torch.isnan(sft_matrix.sum(dim=1))]

        # print('shape of filtered_sft_matrix: {}, shape of filtered_sim_labels {}'.format(filtered_sft_matrix.shape,
        #                                                                                  filtered_sim_labels.shape))
        # print('filtered_sft_matrix: {}, filtered_sim_labels {}'.format(filtered_sft_matrix,
        #                                                                filtered_sim_labels))

        P = (final_mask & fl_sim_labels.bool()).float()
        N = (final_mask & ~(fl_sim_labels.bool())).float()
        strong_pos = (final_mask & fl_sim_labels.bool()).float().sum().item()
        strong_neg = (final_mask & ~(fl_sim_labels.bool())).float().sum().item()
        num_pos = torch.mean(torch.sum((final_mask & fl_sim_labels.bool()), dim=1).float()).item()
        metric_data = (strong_pos, strong_neg, num_mask_ghost, num_pos)

        if (filtered_sft_matrix.shape[0] == 0.):
            return torch.tensor(0., requires_grad=True).float(), metric_data

        num = torch.sum(filtered_sft_matrix * filtered_sim_labels, dim=1)
        den = torch.sum(filtered_sft_matrix, dim=1)
        # print('num {}'.format(num))
        # print('den {}'.format(den))

        # FILTER OUT when all are positives for a source
        mean_loss = -1 * torch.mean(torch.log(num / den))
        
        if self.ss_loss:
            src_src_loss = self.__calc_ss_loss(out_src, src_labels)
            mean_loss += src_src_loss
        
        return mean_loss
