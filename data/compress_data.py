# This script converts x_arrayset.npy (a huge datafile into smaller compressed files)

import gzip
import numpy as np
import os
from filesplit.split import Split
import pdb

# Save as x_arrayset.npy as a compressed file first
data = np.load('x_arrayset.npy',allow_pickle=True)
f = gzip.GzipFile('x_data.npy.gz',"w")
np.save(file=f, arr=data)
f.close()

# split compressed file into smaller files of 20MB each
split = Split('x_data.npy.gz','./')
split.bysize(20971520)

def remove_file(file_path):
	if os.path.exists(file_path):
		os.remove(file_path)
		print(f"The file {file_path} has been deleted.")
	else:
		print(f"The file {file_path} does not exist.")

# delete file
remove_file('x_arrayset.npy')
remove_file('x_data.npy.gz')
