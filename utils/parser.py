'''
Author: Yidan Liu 1334252492@qq.com
Date: 2023-11-11 19:33:19
LastEditors: Yidan Liu 1334252492@qq.com
LastEditTime: 2024-02-26 09:19:33
FilePath: /model1/utils/parser.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='MSDCCL', type=str, help="model selection"
                                                                      "you can choose one of ['MSDCCL', 'GRU4Rec', 'NARM', 'Caser', 'BERT4Rec', 'SASRec', 'STAMP']")
    parser.add_argument('--dataset', default='ml-100k', type=str, help="Dataset to use"
                                                                       "you can choose one of ml-100k, ml-1m, amazon-beauty, amazon-sports-outdoors, yelp")
    parser.add_argument('--sub_model', default='BERT4Rec', type=str, help='sub-model selection'
                                                                               "you can choose one of ['GRU4Rec', 'NARM', 'Caser', 'BERT4Rec', 'SASRec', 'STAMP']")
    # parser.add_argument('--local_rank', default=0, type=int,
    #                     help='node rank for distributed training')

    parser.add_argument('--seed', default=2023, type=int, help='seed for experiment')
    parser.add_argument('--verbose', default=False, type=bool, help='whether save the log file')
    parser.add_argument('--embedding_size', default=100, type=int, help='embedding size for all layer')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="weight decay for adam optimizer")
    parser.add_argument('--train_batch_size', default=256, type=int, help='train batch size')
    parser.add_argument('--is_gumbel_tau_anneal', default=True, type=bool, help='whether use anneal for gumbel')
    parser.add_argument('--gumbel_temperature', default=0.5, type=float, help='gumbel temperature')
    parser.add_argument('--is_spu_cl_tau_anneal', default=True, type=bool, help='whether use anneal for spu_cl')
    parser.add_argument('--supervised_contrastive_temperature', default=0.5, type=float, help='supervised contrastive temperature')
    parser.add_argument('--our_att_drop_out', default=0.5, type=float, help='our attention drop-out rate')
    parser.add_argument('--our_ae_drop_out', default=0.3, type=float, help='our average drop-out rate')
    parser.add_argument('--load_pre_train_emb', default=False, type=bool, help='whether load pre-train model embedding')
    parser.add_argument('--sequence_last_m', default=3, type=int, help='toke the last m items for short interest')
    parser.add_argument('--curriculum_learn_epoch', default=10, type=int, help='the epoch number of using curriculum learning')
    parser.add_argument('--sigmoid_extent', default=5, type=int, help='the sigmoid extent number')
    parser.add_argument('--reweight_loss_alpha', default=0.001, type=float, help='reweight loss alpha')
    parser.add_argument('--reweight_loss_lambda', default=1.0, type=float, help='reweight loss lambda')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')
    parser.add_argument('--warm_up_ratio', default=0, type=int, help='warm up ratio for lr')

    parser.add_argument('--train_set_ratio', default=1, type=float, help='train set ratio for model')

    args = parser.parse_args()
    return args
