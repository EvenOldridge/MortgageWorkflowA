import torch

class BatchDataset(object):
    """An abstract class representing a Batch Dataset.
    All other datasets should subclass this. All subclasses should override
    ``__len__``, which provides the size of the dataset, ``__getitem__``,
    supporting integer indexing of batches in range from 0 to len(self)//batchsize exclusive, 
    and ``shuffle`` which randomly shuffles the data, generally called per epoch.
    Batch datasets are meant to be iterated over in order rather than randomly accessed
    so the randomization has to happen first.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
  
    def shuffle(self):
        raise NotImplementedError

        
        
class TensorBatchDataset(BatchDataset):
    """Batch Dataset wrapping Tensors.  
    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
        batch_size: The size of the batch to return
        
        
    """
    def __init__(self, tensors, batch_size=1):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors      
        self.batch_size=batch_size
        
        self.num_samples = tensors[0].size(0)
    
    def __len__(self):
        if self.num_samples%self.batch_size == 0:
            return self.num_samples // self.batch_size
        else:
            return self.num_samples // self.batch_size + 1

    def __getitem__(self, item):
        idx = item*self.batch_size
        #Need to handle odd sized batches if data isn't divisible by batchsize
        if idx < self.num_samples and (idx + self.batch_size < self.num_samples or self.num_samples%self.batch_size == 0):
            return tuple(tensor[idx:idx+self.batch_size] for tensor in self.tensors)
        elif idx < self.num_samples and idx + self.batch_size> self.num_samples :
            return tuple(tensor[idx:] for tensor in self.tensors)
        else:
            raise IndexError
        
        
    
    def shuffle(self):
        idx = torch.randperm(self.num_samples, dtype=torch.int64)
        self.tensors = tuple(tensor[idx] for tensor in self.tensors)


