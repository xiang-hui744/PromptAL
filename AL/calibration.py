import torch
import ALconfig


def calibration(a: torch.tensor) -> torch.tensor:
    """

Args:
    a: Shape: batch*class probability distribution

Returns: Calibrated probability distribution, Shape: batch*class

    """

    top_k_indices = []
    class_num = a.size(-1)

    print('top k--------')
    for class_idx in range(class_num):
        topk_values, topk_indices = torch.topk(a[:, class_idx], ALconfig.ALargs.k)

        top_k_indices.extend(topk_indices.tolist())

    print('finish top k')

    prior = a[top_k_indices].mean(dim=0)

    outputs_all_normalized = a / prior

    row_sums = outputs_all_normalized.sum(dim=1, keepdim=True)  #
    b = outputs_all_normalized / row_sums  # (batch_size, 5)
    return b
