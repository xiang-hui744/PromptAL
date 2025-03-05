import argparse
import json
import logging
import os
import random
from collections import OrderedDict, defaultdict
from datetime import datetime

from sympy.benchmarks.bench_discrete_log import data_set_1
from transformers import set_seed
from transformers.utils.versions import require_version

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)
def class_labels(dataset):

    path = os.path.join(project_root,f"/data/{dataset}/train.json")
    with open(path, 'r') as f:
        lines = f.readlines()
    train_pool_labels = []
    for i in lines:
        i = json.loads(i)

        train_pool_labels.append(i["_id"])
    return train_pool_labels


class ALConfig:
    def __init__(self,

                 ):


        self.dataset = None  # choices=['dbpedia', 'imdb', 'agnews','yelp','trec','yahoo'])
        self.iterations = 10  # 总AL循环次数
        self.v_tokens_n = 4  #####  task prompt nums
        self.instance_tokens = None  ### ins nums
        self.AL_method = None  # AL 方法
        self.total_trainer = None  # 使用的模型
        self.Q_size = 32  # 冷启动选择的样本数
        self.k_max = 32  # 每轮选择的样本数

        self.dev_size = None  # 验证集的大小
        self.unlabeled_size = None  # 未标记数据的大小
        self.init_train_number = 32  # 初始训练集大小
        self.knn = 10

        self.class_labels = None
        self.unlabeled_embedding = None  # 未标记数据的嵌入表示
        self.h_mask = None
        self.selected_index = []  # 已选择样本索引列表，默认为空列
        self.pool_dataloader = None


        self.pool_ing = False
        self.test_results = defaultdict(list)

        self.current_iterations = 0  # 当前的AL循环次数
        self.foundation_model = None
        self.prior_probability = []  # 记录每种类别的先验概率
        self.k = 100  # 每种类别选k个最高的先验概率样本


        self.label_ids = None

        #### 提示结合方法相关
        self.multi_head = None
        self.prompt_combine_method = 'self attention'


        ### local div
        self.initial_alpha = 0.9
        self.diversity_increase_factor = 0.1
        self.vector = 'hmask' #### hmask cls simcse



    def __repr__(self):
        """
        定义类的字符串表示，方便打印输出所有配置参数。
        """
        return (f"ALConfig(AL_method={self.AL_method}, "
                # f"total_model={self.total_model}, "
                f"k_max ={self.k_max}, "
                f"iterations = {self.iterations}, "
                f"k = {self.k}, "
                f"unlabeled_size={self.unlabeled_size}, "
                f"selected_size={len(self.selected_index)}, "
                f"test result={self.test_results}), "
                f"v_n={self.v_tokens_n}, "
                f"iter={self.iterations}, "
                f"init_n={self.init_train_number}, "
                f"Q_size={self.k_max}, "
                f"k={self.k}, "
                f"knn={self.knn}, "
                f"head_n={self.multi_head}, "
                f"combine={self.prompt_combine_method}, "
                f"final_size={len(self.selected_index)}")

    def get_path(self):
        path_parts = [
            str(self.dataset),
            str(self.AL_method),
            f"task_n{self.v_tokens_n}",
            f"ins_n{self.instance_tokens}",
            f"initial_alpha{self.initial_alpha}",
            f"dstep{self.diversity_increase_factor}",
            f"iter{self.iterations}",
            f"init_n{self.init_train_number}",
            f"Q_size{self.k_max}",
            f"k{self.k}",
            f"knn{self.knn}",
            f"head_n{self.multi_head}",
            f"combine{self.prompt_combine_method}",
            f"final_size{len(self.selected_index)}",
        ]

        # 通过下划线连接所有的部分
        path = "_".join(path_parts) + '.json'
        return path


ALargs = ALConfig()
