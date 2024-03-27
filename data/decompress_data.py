# This script converts smaller compressed files into x_arrayset.npy (a huge datafile)

import gzip
import numpy as np
import os
from filesplit.merge import Merge

# Merge split files
merge = Merge('./', './','x_data.npy.gz')
merge.merge(cleanup = True)

# Decompress .npy.gz file
f = gzip.GzipFile('x_data.npy.gz',"r")
data = np.load(f)
f.close()

# Save data as .npy file
np.save('x_arrayset.npy', arr=data)

def remove_file(file_path):
	if os.path.exists(file_path):
		os.remove(file_path)
		print(f"The file {file_path} has been deleted.")
	else:
		print(f"The file {file_path} does not exist.")

# delete file
remove_file('x_data.npy.gz')
