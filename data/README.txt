Large data files are split into smaller files by running script 'compress_data.py'. Before using the dataset for training, users need to combine the data files by running 'decompress_data.py'.

After running 'decompress_data.py', original x_data.npy_*.gz (with * = 1,2,3,4.....) files will be deleted and a file named 'x_arrayset.npy' will be created.

x_arrayset.npy: 20000*200*200, curvature map (200*200) dataset used for training the inverse and forward model.

y_arrayset.npy: 20000*7, polynomial design parameters used for training forward and inverse model.

id_del.npy: used to transform the 200*200 curvature maps to 1*20100 vectors.

id_save.npy: used to transform the 1*20100 vectors to 200*200 curvature maps.

scaler.pkl: scaling (normalization) data of design parameters.
