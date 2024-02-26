#！/bin/bash

# MSDCCL+BERT4Rec on ml-100k
CUDA_VISIBLE_DEVICES=0 python run_msdccl.py --gpu_id 0 --dataset 'ml-100k' --weight_decay 1e-4 --our_att_drop_out 0.5 --our_ae_drop_out 0.3 --sequence_last_m 3 --reweight_loss_alpha 0.001 --reweight_loss_lambda 1

# MSDCCL+BERT4Rec on amazon-beauty
# CUDA_VISIBLE_DEVICES=0 python run_msdccl.py --gpu_id 1 --dataset 'amazon-beauty' --weight_decay 1e-4 --our_att_drop_out 0.4 --our_ae_drop_out 0.5 --sequence_last_m 2 --reweight_loss_alpha 0.001 --reweight_loss_lambda .2

# MSDCCL+BERT4Rec on amazon-sports-outdoors
# CUDA_VISIBLE_DEVICES=0 python run_msdccl.py --gpu_id 2 --dataset 'amazon-sports-outdoors' --weight_decay 1e-4 --our_att_drop_out 0.3 --our_ae_drop_out 0.7 --sequence_last_m 2 --reweight_loss_alpha 0.001 --reweight_loss_lambda .2

# MSDCCL+BERT4Rec on yelp
# CUDA_VISIBLE_DEVICES=0 python run_msdccl.py --gpu_id 0 --dataset 'yelp' --weight_decay 1e-3 --our_att_drop_out 0 --our_ae_drop_out 0.5 --sequence_last_m 1 --reweight_loss_alpha 0.001 --reweight_loss_lambda .2

# MSDCCL+BERT4Rec on ml-1m
# CUDA_VISIBLE_DEVICES=0 python run_msdccl.py --gpu_id 1 --dataset 'ml-1m' --weight_decay 0 --our_att_drop_out 0.3 --our_ae_drop_out 0.2 --sequence_last_m 6 --reweight_loss_alpha 0.001 --reweight_loss_lambda .6

# dataset
#a = ['amazon-books', 'amazon-toys-games', 'amazon-clothing-shoes-jewelry'
#     'amazon-video-games', 'avazu', 'criteo', 'food', 'netflix']