# py_script="
# import os
# os.chdir('/vulcan-pvc1/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation/')
# "
pip3 install easydict
cd /vulcan-pvc1/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation/
# python3 -c "$py_script"
./experiments/scripts/train.sh ./experiments/config/Office-31/CAN/office31_train_amazon2webcam_cfg.yaml 0 CAN office31_a2w
