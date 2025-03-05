import heapq
import json
import os
import pickle
import random

from AL.calibration import calibration

from ALconfig import ALargs
import torch
import numpy as np
from sklearn.cluster import KMeans

from utils.config import args
from sklearn.neighbors import NearestNeighbors
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)


def has_duplicates(lst):

    return len(lst) != len(set(lst))


def check_common_elements(list1, list2):
    """
   Raise an error if the elements in the two lists are duplicated.
    """
    for element in list1:
        if element in list2:
            raise ValueError(f"Same element found: {element}")


def ou_distance(arr1, arr2):
    """
    Args:
    arr1: Features of sample 1
    arr2: Features of sample 2

Returns: the Euclidean distance.


    """
    return float(np.linalg.norm(arr1 - arr2))


def get_mean_distance(sample_number, hmask_np, train_ids) -> float:
    """
Args:
    sample_number: Number of approximately randomly selected unlabeled samples
    hmask_np: Embedding feature vectors
    train_ids: Indexes of the training set samples

Returns: Average distance β


    """
    len_all = hmask_np.shape[0]
    pool_indices = [i for i in range(len_all) if i not in ALargs.selected_index]
    random.seed(42)
    random_inds = random.sample(pool_indices, sample_number)
    total_dis = 0.0
    for pool_id in random_inds:

        pool_hmask = hmask_np[pool_id]
        for train_id in train_ids:
            train_hmask = hmask_np[train_id]
            dis = ou_distance(pool_hmask, train_hmask)
            total_dis = total_dis + dis
    mean_dis = total_dis / (len(random_inds) * len(train_ids))

    return mean_dis


def get_nearest_distance(all_hmask_np, pool_index, train_index):
    """
   Args:
    all_hmask_np: np features of all samples in the pool
    pool_index: Index of the current sample
    train_index: Index of the training set samples

Returns: the closest distance between the current sample and the training set samples

    """
    pool_hmask = all_hmask_np[pool_index]
    nearst_dis = float('inf')
    for train_id in train_index:
        train_hmask = all_hmask_np[train_id]
        dis = ou_distance(pool_hmask, train_hmask)
        if dis < nearst_dis:
            nearst_dis = dis
    return nearst_dis



def min_max_normalize(data: list):
    min_value = min(data)
    max_value = max(data)
    return [(x - min_value) / (max_value - min_value) for x in data]


def get_knn_average_distance(all_hmask_np, pool_index, train_index, k=10):
    """
    Args:
    all_hmask_np: np features of all samples in the pool
    pool_index: Index of the current sample
    train_index: Index of the training set samples
    k: Number of nearest neighbors
    Returns:
    float: Average distance between the current sample and the k nearest samples from the training set

    """
    pool_hmask = all_hmask_np[pool_index]
    train_hmask = all_hmask_np[train_index]

    knn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='kd_tree')
    knn.fit(train_hmask)

    distances, indices = knn.kneighbors([pool_hmask])

    avg_distance = np.mean(distances)

    return avg_distance


def combine_scores(lista_normalized, listb_normalized, alpha=0.9):
    return alpha * lista_normalized + (1 - alpha) * listb_normalized


def combine_scores_dynamic(lista_normalized, listb_normalized, iteration, max_iterations,
                           initial_alpha=ALargs.initial_alpha,
                           diversity_increase_factor=ALargs.diversity_increase_factor):
    """
    :param lista_normalized: Information score (list)
    :param listb_normalized: Diversity score (list)
    :param iteration: Current active learning iteration
    :param max_iterations: Maximum number of iterations
    :param initial_alpha: Initial alpha value (weight for information score)
    :param diversity_increase_factor: Step size for increasing diversity score weight
    :return: Weighted score

    """


    # alpha = initial_alpha * (1 - (iteration / max_iterations) * diversity_increase_factor)
    alpha = initial_alpha

    combined_score = [alpha * a + (1 - alpha) * b for a, b in zip(lista_normalized, listb_normalized)]

    return combined_score


################################

def entropy():
    softmax = torch.nn.Softmax(dim=1)
    entropy_pool = []

    for logits_original in ALargs.total_trainer.predict_pool():
        # trainer.predict_pool()
        logits = logits_original[:, ALargs.label_ids]
        print("pool batch logits.shape: ", logits.shape)

        outputs = softmax(logits)
        entropy_score = -torch.sum((torch.log(outputs) * outputs), dim=-1)
        entropy_cpu = entropy_score.cpu().numpy().tolist()
        entropy_pool.extend(entropy_cpu)

    print("entropy pool size:", len(entropy_pool))
    available_indices = [i for i in range(len(entropy_pool)) if i not in ALargs.selected_index]

    k_index = heapq.nlargest(
        ALargs.k_max,
        available_indices,
        key=entropy_pool.__getitem__
    )

    assert len(k_index) == ALargs.k_max
    return k_index


def hmask_cluster() -> list[int]:
    """
    Clustering based on hmask vector to select samples (randomly selecting one sample from each cluster after clustering).
    Returns:
    """

    for logits in ALargs.total_trainer.predict_pool():
        pass

    available_indices = [i for i in range(ALargs.unlabeled_size) if i not in ALargs.selected_index]
    hmask = ALargs.h_mask


    hmask_np = hmask.cpu().numpy() if hmask.is_cuda else hmask.numpy()

    hmask_np = hmask_np[available_indices]


    kmeans = KMeans(n_clusters=ALargs.k_max, init='k-means++', random_state=42)
    kmeans.fit(hmask_np)


    labels = kmeans.labels_


    selected_indices = []
    for i in range(ALargs.k_max):

        idx = int(np.where(labels == i)[0][0])
        original_index = available_indices[idx]
        selected_indices.append(original_index)

    return selected_indices


def hmask_cluster_entropy() -> list[int]:
    """
    Clustering based on hmask vector, selecting the sample index with the highest entropy from each cluster.


    """
    softmax = torch.nn.Softmax(dim=1)
    entropy_pool = []

    for logits_original in ALargs.total_trainer.predict_pool():
        logits = logits_original[:, ALargs.label_ids]
        print("pool batch logits.shape: ", logits.shape)

        outputs = softmax(logits)
        entropy_score = -torch.sum((torch.log(outputs) * outputs), dim=-1)
        entropy_cpu = entropy_score.cpu().numpy().tolist()
        entropy_pool.extend(entropy_cpu)

    available_indices = [i for i in range(ALargs.unlabeled_size) if i not in ALargs.selected_index]
    hmask = ALargs.h_mask


    hmask_np = hmask.cpu().numpy() if hmask.is_cuda else hmask.numpy()

    hmask_np = hmask_np[available_indices]

    #  KMeans cluster
    kmeans = KMeans(n_clusters=ALargs.k_max, init='k-means++', random_state=42)
    kmeans.fit(hmask_np)

    labels = kmeans.labels_

    selected_indices = []
    for i in range(ALargs.k_max):

        cluster_indices = np.where(labels == i)[0].tolist()

        entropy_score = 0
        entropy_max_idx = -1
        for indices in cluster_indices:
            pool_index = available_indices[indices]
            if entropy_pool[pool_index] > entropy_score:
                entropy_score = entropy_pool[pool_index]
                entropy_max_idx = pool_index

        original_index = entropy_max_idx
        selected_indices.append(original_index)

    return selected_indices


def hmask_cluster_calibration_entropy() -> list[int]:
    """
        K-means clustering based on hmask vector, selecting the sample index with the highest entropy after calibration for each cluster.


    """
    softmax = torch.nn.Softmax(dim=1)

    available_indices = [i for i in range(ALargs.unlabeled_size) if i not in ALargs.selected_index]

    outputs_all = None

    for logits_original in ALargs.total_trainer.predict_pool():

        logits = logits_original[:, ALargs.label_ids]
        print("pool batch logits.shape: ", logits.shape)

        outputs = softmax(logits)
        if outputs_all is None:
            outputs_all = outputs
        else:
            outputs_all = torch.cat((outputs_all, outputs), dim=0)

    outputs_normalized_final = calibration(outputs_all)

    entropy_score = -torch.sum((torch.log(outputs_normalized_final) * outputs_normalized_final), dim=-1)
    entropy_pool = entropy_score.cpu().numpy().tolist()
    del outputs_normalized_final
    del outputs_all, outputs

    hmask = ALargs.h_mask

    hmask_np = hmask.cpu().numpy() if hmask.is_cuda else hmask.numpy()

    del hmask
    hmask_np = hmask_np[available_indices]

    kmeans = KMeans(n_clusters=ALargs.k_max, init='k-means++', random_state=42)
    kmeans.fit(hmask_np)

    labels = kmeans.labels_

    selected_indices = []
    for i in range(ALargs.k_max):

        cluster_indices = np.where(labels == i)[0].tolist()


        entropy_score = 0
        entropy_max_idx = -1
        for indices in cluster_indices:
            pool_index = available_indices[indices]
            if entropy_pool[pool_index] > entropy_score:
                entropy_score = entropy_pool[pool_index]
                entropy_max_idx = pool_index

        original_index = entropy_max_idx
        selected_indices.append(original_index)

    return selected_indices


def hmask_cluster_calibration_entropy_local_div() -> list[int]:
    """     DSPAL method
            Returns: Selected sample indices

            """
    softmax = torch.nn.Softmax(dim=1)

    available_indices = [i for i in range(ALargs.unlabeled_size) if i not in ALargs.selected_index]

    outputs_all = None

    for logits_original in ALargs.total_trainer.predict_pool():

        logits = logits_original[:, ALargs.label_ids]
        print("pool batch logits.shape: ", logits.shape)

        outputs = softmax(logits)
        if outputs_all is None:
            outputs_all = outputs
        else:
            outputs_all = torch.cat((outputs_all, outputs), dim=0)

    #### calibration
    outputs_normalized_final = calibration(outputs_all)

    #### entropy
    entropy_score = -torch.sum((torch.log(outputs_normalized_final) * outputs_normalized_final), dim=-1)
    entropy_pool = entropy_score.cpu().numpy().tolist()
    # del outputs_normalized_final
    del outputs_all, outputs


    if ALargs.vector == 'simcse':
        out_path = os.path.join(project_root, f"data/{ALargs.dataset}/train_simcse.pkl")
        print('simcse cluster--------')
        with open(out_path, 'rb') as f:

            hmask_np_all = pickle.load(f)

    else:
        hmask = ALargs.h_mask

        hmask_np_all = hmask.cpu().numpy() if hmask.is_cuda else hmask.numpy()

        del hmask
    hmask_np = hmask_np_all[available_indices]

    print('Kmeans ---')

    kmeans = KMeans(n_clusters=ALargs.k_max, init='k-means++', random_state=42)
    kmeans.fit(hmask_np)
    print('finish Kmeans ---')

    labels = kmeans.labels_

    selected_indices = []

    # mean_dis = get_mean_distance(1000,hmask_np_all,ALargs.selected_index)

    for i in range(ALargs.k_max):
        ###
        cluster_indices = np.where(labels == i)[0].tolist()

        ######
        pool_cluster_index = [available_indices[indices] for indices in cluster_indices]
        entropy_cluster_score_list = [entropy_pool[i] for i in pool_cluster_index]
        dis_cluster_score_list = []

        ###### KNN
        # print('knn------')
        for index in pool_cluster_index:
            avg_distance = get_knn_average_distance(hmask_np_all, index, ALargs.selected_index, k=ALargs.knn)
            dis_cluster_score_list.append(avg_distance)

        # Min-Max
        # entropy_cluster_score_normalize = min_max_normalize(entropy_cluster_score_list)
        dis_cluster_score_normalize = min_max_normalize(dis_cluster_score_list)

        # entropy_cluster_score_normalize = z_score_normalize(entropy_cluster_score_list)
        # dis_cluster_score_normalize = z_score_normalize(dis_cluster_score_list)

        total_scores = combine_scores_dynamic(entropy_cluster_score_list,
                                              dis_cluster_score_normalize,
                                              ALargs.current_iterations,
                                              ALargs.iterations,
                                              )
        max_index = total_scores.index(max(total_scores))

        selected_index = pool_cluster_index[max_index]

        selected_indices.append(selected_index)

    return selected_indices


def hmask_cluster_entropy_local_div() -> list[int]:
    """
                Ablation method: No probability calibration

                Returns: Selected sample indices

                """
    softmax = torch.nn.Softmax(dim=1)

    available_indices = [i for i in range(ALargs.unlabeled_size) if i not in ALargs.selected_index]

    outputs_all = None

    for logits_original in ALargs.total_trainer.predict_pool():

        logits = logits_original[:, ALargs.label_ids]
        print("pool batch logits.shape: ", logits.shape)

        outputs = softmax(logits)
        if outputs_all is None:
            outputs_all = outputs
        else:
            outputs_all = torch.cat((outputs_all, outputs), dim=0)

    outputs_normalized_final = outputs_all

    #### entropy
    entropy_score = -torch.sum((torch.log(outputs_normalized_final) * outputs_normalized_final), dim=-1)
    entropy_pool = entropy_score.cpu().numpy().tolist()
    del outputs_all, outputs

    hmask = ALargs.h_mask

    hmask_np_all = hmask.cpu().numpy() if hmask.is_cuda else hmask.numpy()

    del hmask
    hmask_np = hmask_np_all[available_indices]

    kmeans = KMeans(n_clusters=ALargs.k_max, init='k-means++', random_state=42)
    kmeans.fit(hmask_np)

    labels = kmeans.labels_

    selected_indices = []

    for i in range(ALargs.k_max):

        cluster_indices = np.where(labels == i)[0].tolist()


        pool_cluster_index = [available_indices[indices] for indices in cluster_indices]
        entropy_cluster_score_list = [entropy_pool[i] for i in pool_cluster_index]
        dis_cluster_score_list = []


        for index in pool_cluster_index:
            avg_distance = get_knn_average_distance(hmask_np_all, index, ALargs.selected_index, k=ALargs.knn)
            dis_cluster_score_list.append(avg_distance)

        dis_cluster_score_normalize = min_max_normalize(dis_cluster_score_list)

        total_scores = combine_scores_dynamic(entropy_cluster_score_list,
                                              dis_cluster_score_normalize,
                                              ALargs.current_iterations,
                                              ALargs.iterations,
                                              )
        max_index = total_scores.index(max(total_scores))

        selected_index = pool_cluster_index[max_index]

        selected_indices.append(selected_index)

    return selected_indices


def hmask_cluster_local_div() -> list[int]:

    softmax = torch.nn.Softmax(dim=1)

    available_indices = [i for i in range(ALargs.unlabeled_size) if i not in ALargs.selected_index]

    outputs_all = None

    for logits_original in ALargs.total_trainer.predict_pool():

        logits = logits_original[:, ALargs.label_ids]
        print("pool batch logits.shape: ", logits.shape)

        outputs = softmax(logits)
        if outputs_all is None:
            outputs_all = outputs
        else:
            outputs_all = torch.cat((outputs_all, outputs), dim=0)

    del outputs_all, outputs

    #### cluster related
    hmask = ALargs.h_mask

    hmask_np_all = hmask.cpu().numpy() if hmask.is_cuda else hmask.numpy()

    del hmask
    hmask_np = hmask_np_all[available_indices]

    #  KMeans
    kmeans = KMeans(n_clusters=ALargs.k_max, init='k-means++', random_state=42)
    kmeans.fit(hmask_np)

    labels = kmeans.labels_
    selected_indices = []

    for i in range(ALargs.k_max):

        cluster_indices = np.where(labels == i)[0].tolist()
        pool_cluster_index = [available_indices[indices] for indices in cluster_indices]

        dis_cluster_score_list = []

        for index in pool_cluster_index:
            avg_distance = get_knn_average_distance(hmask_np_all, index, ALargs.selected_index, k=ALargs.knn)
            dis_cluster_score_list.append(avg_distance)
        dis_cluster_score_normalize = min_max_normalize(dis_cluster_score_list)
        total_scores = dis_cluster_score_normalize
        max_index = total_scores.index(max(total_scores))

        selected_index = pool_cluster_index[max_index]

        selected_indices.append(selected_index)



    return selected_indices


def simcse_cluster_calibration_entropy() -> list[int]:

    out_path = os.path.join(project_root, f"data/{args.dataset_name}/unlabeled_simcse.pkl")

    with open(out_path, 'rb') as f:

        data = pickle.load(f)


    available_indices = [i for i in range(ALargs.unlabeled_size) if i not in ALargs.selected_index]

    hmask_np = data[available_indices]


    softmax = torch.nn.Softmax(dim=1)

    outputs_all = None

    for logits_original in ALargs.total_trainer.predict_pool():

        logits = logits_original[:, ALargs.label_ids]
        print("pool batch logits.shape: ", logits.shape)

        outputs = softmax(logits)
        if outputs_all is None:
            outputs_all = outputs
        else:
            outputs_all = torch.cat((outputs_all, outputs), dim=0)

    outputs_normalized_final = calibration(outputs_all)


    entropy_score = -torch.sum((torch.log(outputs_normalized_final) * outputs_normalized_final), dim=-1)
    entropy_pool = entropy_score.cpu().numpy().tolist()
    del outputs_normalized_final
    del outputs_all, outputs

    kmeans = KMeans(n_clusters=ALargs.k_max, init='k-means++', random_state=42)
    kmeans.fit(hmask_np)

    labels = kmeans.labels_

    selected_indices = []
    for i in range(ALargs.k_max):

        cluster_indices = np.where(labels == i)[0].tolist()

        entropy_score = 0
        entropy_max_idx = -1
        for indices in cluster_indices:
            pool_index = available_indices[indices]
            if entropy_pool[pool_index] > entropy_score:
                entropy_score = entropy_pool[pool_index]
                entropy_max_idx = pool_index

        original_index = entropy_max_idx
        selected_indices.append(original_index)

    return selected_indices


def init_trainset(num) -> list[int]:
    with open(os.path.join(project_root, f'data/{ALargs.dataset}/init_train_index.json'), 'r', encoding='utf-8') as f:
        index_train = json.load(f)

    print('init train size：', len(index_train))
    assert len(index_train) == num
    return index_train


def find_examples() -> list[int]:
    """
    Returns:
        list[int]: AL selected data index
    """

    filtered_pool_index = [idx for idx in range(ALargs.unlabeled_size) if idx not in ALargs.selected_index]
    index = []

    if len(filtered_pool_index) < ALargs.k_max:
        print(f"not enough {ALargs.k_max} ,only {len(filtered_pool_index)} ")

    if ALargs.current_iterations == 0:

        index = random.sample(filtered_pool_index, ALargs.k_max)
    else:
        if ALargs.AL_method == 'random':
            index = random.sample(filtered_pool_index, ALargs.k_max)

        if ALargs.AL_method == 'entropy':
            index = entropy()

        if ALargs.AL_method == 'hmask_cluster':
            index = hmask_cluster()

        if ALargs.AL_method == 'hmask_cluster_entropy':
            index = hmask_cluster_entropy()

        if ALargs.AL_method == 'hmask_cluster_calibration_entropy':
            index = hmask_cluster_calibration_entropy()

        if ALargs.AL_method == 'hmask_cluster_calibration_entropy_local_div':
            index = hmask_cluster_calibration_entropy_local_div()
            # index = hmask_cluster_calibration_entropy()

        if ALargs.AL_method == 'hmask_cluster_entropy_local_div':
            #### remove calibration
            index = hmask_cluster_entropy_local_div()

        if ALargs.AL_method == 'hmask_cluster_local_div':
            #### remove entropy
            index = hmask_cluster_local_div()


        if ALargs.AL_method == 'simcse_cluster_calibration_entropy':
            index = simcse_cluster_calibration_entropy()

    if has_duplicates(index):
        raise ValueError("The list has duplicates!")
    check_common_elements(index, ALargs.selected_index)

    return index
