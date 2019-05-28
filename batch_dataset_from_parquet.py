import cudf, torch
from torch.utils import data as torch_data
from torch.utils.dlpack import from_dlpack
import glob, os
import numpy as np
import pyarrow.parquet as pq
import pdb
import batch_dataset

# Load parquet file during init

def load_tensors_from_parquet_via_cudf(path, target_name='delinquency_12'):
    gdf = cudf.read_parquet(path)
    target = None
    if target_name in gdf.columns:
        target = from_dlpack(gdf[target_name].astype('float32').to_dlpack())
    # hack because we can't cast a whole dataframe
    for col in gdf.columns:
        gdf[col] = gdf[col].astype('int64')
    tensors = from_dlpack(gdf[:].drop(target_name).to_dlpack())
    # if target is not None:
    #    tensors.append(target)
    return tensors, target

def load_tensors_from_parquet(path, target_name='delinquency_12'):
    tbl = pq.read_table(path).to_pandas()
    target = None
    if target_name in tbl:
        target = torch.from_numpy(tbl.pop(target_name).values.astype(np.float32))
    tensors = torch.from_numpy(tbl.values.astype(np.long))
    
    return tensors, target


def parquet_to_tensor (root_path, target_name='delinquency_12', num_files=None, file_offset=0, use_cuDF=False, use_GPU_RAM=False):
    """Reads in a directory worth of parquet files, returning a tensor containing the features and targets"""
    parquet_files = glob.glob(os.path.join(root_path, "*.parquet"))

    num_files = len(parquet_files) if num_files is None else num_files
    
    i = 0
    targets = None
    features = None
    for f in parquet_files:
        if i >= file_offset:
            if i >= num_files+file_offset: break
            if use_cuDF:
                feature, target = load_tensors_from_parquet_via_cudf(f)
            else: 
                feature, target = load_tensors_from_parquet(f)
            if targets is None:
                targets = target
            else:
                targets = torch.cat((targets, target))
            if features is None:
                features = feature
            else:
                features = torch.cat((features, feature))
            i = i + 1
        else:
            i = i + 1

#         if use_cuDF is False and use_GPU_RAM is True:
#             features = features.cuda()
#             targets = targets.cuda()
#         elif use_cuDF is True and use_GPU_RAM is False:
#             features = features.cpu()
#             targets = targets.cpu()
       

    return (features,targets)
    
def batch_dataset_from_parquet(root_path, num_files=1, file_offset=0, use_cuDF=False, use_GPU_RAM=False, batch_size=1):
    tensors = parquet_to_tensor(root_path, num_files=num_files, file_offset=file_offset, use_cuDF=use_cuDF, use_GPU_RAM=use_GPU_RAM)
    return batch_dataset.TensorBatchDataset(tensors, batch_size)