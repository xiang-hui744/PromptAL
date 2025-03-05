import json
import json
from ALconfig import ALargs


def best_acc_f1(data, seed):
    """

    Args:
        json_path: data

    Returns:best acc dict 和 best f1 dict

    """
    acc_dict = {}
    f1_dict = {}
    for key, value in data.items():
        parts = key.split('_')
        if len(parts) == 3:
            iter_num = int(parts[0])
            metric = parts[2]
        elif len(parts) == 2:
            iter_num = int(parts[0])
            metric = parts[1]
        else:
            continue
        metric_value = round(value[0], 4)
        if 'acc' in metric:
            if iter_num not in acc_dict or metric_value > acc_dict[iter_num]:
                acc_dict[iter_num] = metric_value
        elif 'f1' in metric:
            if iter_num not in f1_dict or metric_value > f1_dict[iter_num]:
                f1_dict[iter_num] = metric_value
    seed_acc_dict = {seed: acc_dict}
    seed_f1_dict = {seed: f1_dict}
    return seed_acc_dict, seed_f1_dict
