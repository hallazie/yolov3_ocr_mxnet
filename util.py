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
	if x0>x1 or y0>y1:
		return 0
	intersec = area(((x0,y0), (x1,y1)))
	return intersec / float(area(vec1)+area(vec2)-intersec)

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

def label_2_bbox(raw_shape, input_shape, label, anchors, num_class, min_down_scale, stride_down_scale):
	'''
		input shape would be 3 anchors each reshaped to a 2-dimension matrix and concat to each other
		thus is easy to process and reverse back to bbox --- false
		input shape would be the original output
	'''
	w_ratio, h_ratio = float(raw_shape[0])/input_shape[0], float(raw_shape[1])/input_shape[1]
	grid_shape = [np.array(input_shape)//{0:min_down_scale, 1:min_down_scale*stride_down_scale, 2:min_down_scale*(stride_down_scale**2)}[i] for i in range(3)] # [array([80, 60]), array([40, 30]), array([20, 15])]
	bbox = {}
	for layer in range(len(grid_shape)):
		cur_shape = grid_shape[layer]
		for x in range(cur_shape[0]):
			for y in range(cur_shape[1]):
				for anchors in range(3):
					if label[layer][x,y,anchors,4] > 0:
						x0, y0 ,x1, y1 = label[layer][x,y,anchors,0:4]
						x0, y0 ,x1, y1 = round(x0*w_ratio), round(y0*h_ratio) ,round(x1*w_ratio), round(y1*h_ratio)
						idx = list(label[layer][x,y,anchors,5:]).index(1.)
						bbox[bbox_dict_reverse[idx]] = [[x0,y0],[x1,y1]]
	return bbox

def bbox_2_label(raw_shape, input_shape, bbox_json, anchors, num_class, min_down_scale, stride_down_scale):
	'''
		for input_shape=(640,640), len(anchors)=3*3, num_class=16, min_down_scale=8
		label shape = [(20,20,3,21),(40,40,3,21),(80,80,3,21)]
		when calculating loss, label will be reshaped to ((80*80*3+40*40*3+20*20*3),21) = (25200,21)

		calculating the max IOU to decide which anchor is the ground truth anchor for each bbox
	'''
	anchors_mask = [[i*3+j for j in range(3)] for i in range(3)]
	grid_shape = [np.array(input_shape)//{0:min_down_scale, 1:min_down_scale*stride_down_scale, 2:min_down_scale*(stride_down_scale**2)}[i] for i in range(3)] # [array([80, 60]), array([40, 30]), array([20, 15])]
	label = [np.zeros((grid_shape[i][0],grid_shape[i][1],3,5+num_class)) for i in range(3)]
	sum_row = np.array([0 for i in range(33)], dtype='float64')
	w_ratio, h_ratio = float(input_shape[0])/raw_shape[0], float(input_shape[1])/raw_shape[1]
	anchors = [(a[0]*w_ratio, a[1]*h_ratio) for a in anchors]
	for bk in bbox_json:
		class_idx = bbox_dict[bk]
		bbox = bbox_json[bk]
		bbox = ((bbox[0][0]*w_ratio, bbox[0][1]*h_ratio), (bbox[1][0]*w_ratio, bbox[1][1]*h_ratio))
		cur_max_iou, cur_max_x, cur_max_y, cur_max_l, cur_max_a = 0, 0, 0, 0, 0
		for layer in range(3):
			for anchor in range(3):
				for x in range(grid_shape[layer][0]):
					for y in range(grid_shape[layer][1]):
						up = min_down_scale*(2**layer)
						x0, y0 = x*up+up/2-anchors[anchors_mask[layer][anchor]][0]/2., y*up+up/2-anchors[anchors_mask[layer][anchor]][1]/2.
						x1, y1 = x*up+up/2+anchors[anchors_mask[layer][anchor]][0]/2., y*up+up/2+anchors[anchors_mask[layer][anchor]][1]/2.
						iou = calculate_iou(((x0,y0),(x1,y1)), bbox)
						if iou >= cur_max_iou:
							cur_max_iou = iou
							cur_max_x = x
							cur_max_y = y
							cur_max_l = layer
							cur_max_a = anchor
		# cur_row = [0,0]+[0,0]+[0]+[1 if i==class_idx else 0 for i in range(num_class)]
		cur_row = list(bbox[0])+list(bbox[1])+[1]+[1 if i==class_idx else 0 for i in range(num_class)]
		label[cur_max_l][cur_max_x, cur_max_y, cur_max_a] =+ np.array(cur_row)
		sum_row += label[cur_max_l][cur_max_x, cur_max_y, cur_max_a]
	return label

if __name__ == '__main__':
	with open('data/jsons/0.json', 'r') as jf:
		bj = json.load(jf)
	anchors = gen_anchor_from_cluster()
	res = bbox_2_label(raw_shape=(1498,955), input_shape=(640,480), bbox_json=bj, anchors=anchors, num_class=28, min_down_scale=8, stride_down_scale=2)
	box = label_2_bbox(raw_shape=(1498,955), input_shape=(640,480), label=res, anchors=anchors, num_class=28, min_down_scale=8, stride_down_scale=2)
	with open('data/tmp.json', 'w') as jf:
		json.dump(box, jf)