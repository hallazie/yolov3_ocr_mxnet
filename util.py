#coding:utf-8

from PIL import Image
from sklearn.cluster import *

from config import *

import json
import os
import numpy as np

def bbox_2_label():
	pass

def gen_anchor_from_cluster(scale=3, num_anchor=3):
	coords = []
	for _,_,fs in os.walk(json_path):
		for f in fs:
			with open(json_path+f, 'r') as jfile:
				dat = json.load(jfile)
				for k in dat:
					c0, c1 = dat[k]
					w, h = c1[0]-c0[0], c1[1]-c0[1]
					coords.append((w,h))
	anchors = KMeans(
		n_clusters=scale*num_anchor,
		init='k-means++',
		precompute_distances='auto',
		copy_x=True,
		algorithm='auto'
		).fit(np.array(coords)).cluster_centers_
	anchors = sorted([tuple((int(e[0]), int(e[1]))) for e in anchors], key=lambda x:x[0]*x[1])
	return anchors

if __name__ == '__main__':
	print gen_anchor_from_cluster()