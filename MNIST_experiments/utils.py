import torch



def gpu_check(self):
    """[Check if GPU is available]

        Returns:
            [obj]: [torch.cuda.is_available()]
    """
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    return train_on_gpu

