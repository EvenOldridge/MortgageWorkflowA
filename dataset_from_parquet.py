import cudf, torch
from torch.utils import data as torch_data
from torch.utils.dlpack import from_dlpack
import glob, os
import numpy as np
import pyarrow.parquet as pq

def load_tensors_from_parquet(path, target_name='delinquency_12'):
    tbl = pq.read_table(path).to_pandas()
    target = None
    if target_name in tbl:
        target = torch.from_numpy(tbl.pop(target_name).values.astype(np.float32))
    features = torch.from_numpy(tbl.values.astype(np.long))
    tensors = [features]
    if target is not None:
        tensors.append(target)
    return tuple(tensors)


class MortgageParquetDataset(torch_data.Dataset):

    def __init__(self, root_path, num_samples=None, target_name='delinquency_12',
                 shuffle_files=False):
        self.parquet_files = glob.glob(os.path.join(root_path, "*.parquet"))
        if shuffle_files:
            self.parquet_files = list(np.random.permutation(self.parquet_files))
        self.target_name = target_name
        self.metadata = [pq.read_metadata(p) for p in self.parquet_files]
        self.cumsum_rows = np.cumsum([m.num_rows for m in self.metadata])

        self.times_through = 0
        if num_samples is not None:
            self.num_samples = min(num_samples, self.cumsum_rows[-1])
        else:
            self.num_samples = self.cumsum_rows[-1]

        self.loaded_tensors = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        tt = self.times_through
        if item == len(self) - 1:
            self.times_through += 1
        item += tt * len(self)
        item %= len(self)

        part_idx = np.searchsorted(self.cumsum_rows, item, side='right')

        if self.loaded_tensors is None or self.loaded_tensors[0] != part_idx:
            tensors = load_tensors_from_parquet(self.parquet_files[part_idx])
            self.loaded_tensors = (part_idx, tensors)

        i = item if part_idx == 0 else item - self.cumsum_rows[part_idx - 1]
        return tuple(tensor[i] for tensor in self.loaded_tensors[1])



def dataset_from_parquet(root_path, num_samples=None, shuffle_files=False):
    return MortgageParquetDataset(root_path, num_samples=num_samples, shuffle_files=shuffle_files)