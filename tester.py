import h5py
import numpy as np
from PIL import Image
import pickle
import os
import torch

def resizer(path):
	hdf = h5py.File(path, 'r')
	array = hdf['dataset'][:]
	array*=256
	img = Image.fromarray(array.astype('uint8'), 'RGB')
	img.save("test-normal-r.jpeg", "JPEG")

#resizer('../normals-r/ai_001_003/images/scene_cam_00_geometry_hdf5/frame.0004.normal_cam.hdf5')

with open('normal.pickle', 'rb') as file:
	temp = pickle.load(file)

total = 0
nans = 0

for i in temp:
	if os.path.exists('../normals-r/' + i):
		_label = torch.from_numpy(h5py.File('../normals-r/' + i, 'r')['dataset'][:].astype('float32'))
		total += 1
		if _label.isnan().any():
			nans += 1

print('NaN percentage: ' + str(nans/total*100) + '%')