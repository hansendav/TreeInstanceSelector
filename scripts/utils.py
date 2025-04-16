import subprocess
from pathlib import Path 
import h5py
import numpy as np


def list_recursive_files(data_dir, ext='h5'): 
    assert Path(data_dir).exists(), f"Data directory {data_dir} does not exist."
    assert Path(data_dir).is_dir(), f"Data directory {data_dir} is not a directory."

    files  = subprocess.check_output(f"find {data_dir} -type f -name '*.{ext}'", shell=True, text=True)
    files = files.strip().split('\n')

    return files

def read_h5(file_path): 
    assert Path(file_path).exists(), f"File {file_path} does not exist."
    assert Path(file_path).is_file(), f"File {file_path} is not a file."

    with h5py.File(file_path, 'r') as f: 
        instances = f['train/clouds'][:]

        instances = [np.reshape(instance, (-1, 3)) for instance in instances]            
    return instances
