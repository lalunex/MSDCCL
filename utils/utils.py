import importlib
import math
import numpy as np


def get_model(model_name):
    model_file_name = model_name.lower()
    module_path = '.'.join(['model', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)
    else:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))

    model_class = getattr(model_module, model_name)

    return model_class


def is_msdccl_model(model_name):
    return 'MSDCCL' in model_name


def compute_indicator(predict, ground_true):
    Hits_i = 0
    MRR_i = 0
    HR_i = 0
    NDCG_i = 0
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            if ground_true[i][0]==predict[i][j]:
                Hits_i+=1
                HR_i+=1
                # 注意j的取值从0开始
                MRR_i+=1/(j+1)   
                NDCG_i+=1/(math.log2(1+j+1))
                break
    return HR_i, NDCG_i, MRR_i

def indicators_20(predict, ground_true, seq_length):
    different_seq_res = {}
    Len_T = len(ground_true)

    # ml-100k, 序列长度为18-50，所以分为0-28， 29-49， 50
    # beauty, 序列长度为4-50，所以分为4， 5-6， 7-50
    # sports，所以分为4， 5-6， 7-50
    # yelp, 所以分为4， 5-7， 8-50
    # ml-1m, 所以分为0-54， 55-167， 167-200
    short_predict_seq = predict[np.where(seq_length <= 4)]
    short_ground_true = np.expand_dims(ground_true[np.where(seq_length <= 4)], axis=-1)
    HR_s, NDCG_s, MRR_s = compute_indicator(short_predict_seq, short_ground_true)
    different_seq_res['short_HR_20'] = HR_s / Len_T
    different_seq_res['short_NDCG_20'] = NDCG_s / Len_T
    different_seq_res['short_MRR_20'] = MRR_s / Len_T

    medium_predict_seq = predict[np.where((seq_length >= 5) &  (seq_length <= 6))]
    medium_ground_true = np.expand_dims(ground_true[np.where((seq_length >= 5) & (seq_length <= 6))], axis=-1)
    HR_m, NDCG_m, MRR_m = compute_indicator(medium_predict_seq, medium_ground_true)
    different_seq_res['medium_HR_20'] = HR_m / Len_T
    different_seq_res['medium_NDCG_20'] = NDCG_m / Len_T
    different_seq_res['medium_MRR_20'] = MRR_m / Len_T

    long_predict_seq = predict[np.where(seq_length >= 7)]
    long_ground_true = np.expand_dims(ground_true[np.where(seq_length >= 7)], axis=-1)
    HR_l, NDCG_l, MRR_l = compute_indicator(long_predict_seq, long_ground_true)
    different_seq_res['long_HR_20'] = HR_l / Len_T
    different_seq_res['long_NDCG_20'] = NDCG_l / Len_T
    different_seq_res['long_MRR_20'] = MRR_l / Len_T

    return different_seq_res
