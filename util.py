#coding:utf-8

from PIL import Image
from sklearn.cluster import *

from config import *

import json
import os
import numpy as np

def area(coords):
	return (coords[1][0]-coords[0][0])*(coords[1][1]-coords[0][1])

def calculate_iou(vec1, vec2):
	'''
		input ((x0,y0), (x1,y1)), ((x2,y2), (x3,y3))
		return float
	'''
	x0, y0 = max(vec1[0][0], vec2[0][0]), max(vec1[0][1], vec2[0][1])
	x1, y1 = min(vec1[1][0], vec2[1][0]), min(vec1[1][1], vec2[1][1])
	intersec = area(((x0,y0), (x1,y1)))
	return intersec / float(area(vec1)+area(vec2)-intersec)

def bbox_2_label(input_shape, bbox_json, anchors, num_class, min_down_scale):
	'''
		for input_shape=(640,640), len(anchors)=3*3, num_class=16, min_down_scale=8
		label shape = [(20,20,3,21),(40,40,3,21),(80,80,3,21)]
		when calculating loss, label will be reshaped to ((80*80*3+40*40*3+20*20*3),21) = (25200,21)

		calculating the max IOU to decide which anchor is the ground truth anchor for each bbox
	'''
	anchors_mask = [[i*3+j for j in range(3)] for i in range(3)]
	grid_shape = [input_shape//{0:min_down_scale, 1:min_down_scale*2, 2:min_down_scale*4}[i] for i in range(3)] # [array([80, 60]), array([40, 30]), array([20, 15])]
	label = [np.zeros((grid_shape[i][0],grid_shape[i][1],3,5+num_class)) for i in range(3)]

	for bk in bbox_json:
		class_idx = bbox_dict[bk]
		bbox = bbox_json[bk]
		cur_max_iou, cur_max_x, cur_max_y, cur_max_l, cur_max_a = 0, 0, 0, 0, 0
		for layer in range(3):
			for anchor in range(3):
				for x in range(grid_shape[layer][0]):
					for y in range(grid_shape[layer][1]):
						up = min_down_scale*(2**layer)
						x0, y0 = x*up+up/2-anchors[anchors_mask[i],[j]][0]/2., y*up+up/2-anchors[anchors_mask[i],[j]][1]/2.
						x1, y1 = x*up+up/2+anchors[anchors_mask[i],[j]][0]/2., y*up+up/2+anchors[anchors_mask[i],[j]][1]/2.
						iou = calculate_iou(((x0,y0),(x1,y1)), bbox)
						if iou > cur_max_iou:
							cur_max_iou = iou
							cur_max_x = x
							cur_max_y = y
							cur_max_l = layer
							cur_max_a = anchor
		cur_row = bbox[0]+bbox[1]+[1]+[1 if i==class_idx else 0 for i in range(num_class)]
		label[cur_max_a][cur_max_x, cur_max_y, cur_max_l] = np.array(cur_row)
	return label

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
	anchors = sorted([tuple((e[0], e[1])) for e in anchors], key=lambda x:x[0]*x[1])
	return anchors

if __name__ == '__main__':
	# anchor_cluster = gen_anchor_from_cluster()
	# anchors_mask = [[i*3+j for j in range(3)] for i in range(3)]
	# for i in range(3):
	# 	for j in range(3):
	# 		print anchor_cluster[anchors_mask[i][j]]
	# 		print anchors_mask[i][j]
	# 		print '-----------------'

	# input_shape = np.array((640,480))
	# min_down_scale = 8
	# num_class = 16
	# grid_shape = [input_shape//{0:min_down_scale, 1:min_down_scale*2, 2:min_down_scale*4}[i] for i in range(3)]
	# label = [np.zeros((grid_shape[i][0],grid_shape[i][1],3,5+num_class)) for i in range(3)]
	# print label[1].shape

	print [1 if i==2 else 0 for i in range(16)]