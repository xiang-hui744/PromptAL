
import torch
from torch import tensor

from ALconfig import ALargs


def get_hmask(last_hidden_state: tensor, mask_indices: tensor)-> tensor:
    """
    Obtain the hmask information for this batch and add it to ALargs' total h_mask.
    Returns: ALargs
    """


    batch_size = last_hidden_state.size(0)
    # seq_length = last_hidden_state.size(1)
    hidden_size = last_hidden_state.size(2)

    indices = mask_indices.view(batch_size, 1, 1).expand(batch_size, 1, hidden_size)

    mask_hidden_states = torch.gather(last_hidden_state, dim=1, index=indices).squeeze(1)

    return mask_hidden_states


def get_cls(last_hidden_state: tensor)-> tensor:



    batch_size = last_hidden_state.size(0)
    # seq_length = last_hidden_state.size(1)
    hidden_size = last_hidden_state.size(2)




    return last_hidden_state[:,0,:]

























