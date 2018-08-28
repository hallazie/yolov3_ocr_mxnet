#coding:utf-8

import util
import os
import mxnet as mx
import numpy as np
import json

from config import *
from PIL import Image

def diter():
	data_path = 'data/imgs/'
	label_path = 'data/jsons/'
	anchor = util.gav_anchor()
	for _,_, fs in os.walk(data_path):
		data = np.zeros((len(fs), WIDTH, HEIGHT, 3))
		label = np.zeros((len(fs), WIDTH//DOWNSAMPLE, HEIGHT//DOWNSAMPLE, 33))
		for i, f in enumerate(fs):
			img = Image.open(data_path+f).resize((WIDTH, HEIGHT), resample=Image.LANCZOS)
			with open(label_path+f.split('.')[0]+'.json', 'r') as lj:
				bbox = json.load(lj)
				res = util.bbox_2_label(raw_shape=img.size, input_shape=(WIDTH,HEIGHT), bbox_json=bbox, anchor=anchor, num_class=28, downscale=16)
			data[i,:,:,:] = mx.nd.array(img).transpose()
			label[i,:,:,:] = res
	return mx.io.NDArrayIter(data=data, label=label, batch_size=BATCH_SIZE, shuffle=True)