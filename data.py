import numpy as np
import datasets as high_dim_data
import torch

def load_data(name):

    if name == 'bsds300':
        return high_dim_data.BSDS300()

    elif name == 'power':
        return high_dim_data.POWER()

    elif name == 'gas':
        return high_dim_data.GAS()

    elif name == 'miniboone':
        return high_dim_data.MINIBOONE()

    else:
        raise ValueError('Unknown dataset')


def tensor_high_dim(name):
    # Same in OT-Flow
    data = load_data(name)
    train_full, test_full = torch.from_numpy(data.trn.x), torch.from_numpy(data.val.x)
    print(f'Train X shape {train_full.shape}')
    print(f'Test X shape {test_full.shape}')
    return train_full, test_full
