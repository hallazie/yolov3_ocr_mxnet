#coding:utf-8

from PIL import Image, ImageDraw
from sklearn.cluster import *

from config import *

import json
import os
import numpy as np
import math

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

def sigmoid(x):
	return 1/(1+e**(-1*x))

def rect(pannel, bbox, outline=(224,32,128)):
	draw = ImageDraw.Draw(pannel)
	w, h = pannel.size
	for kv in bbox:
		pos0, pos1 = bbox[kv]
		x0, y0, x1, y1 = pos0[0], pos0[1], pos1[0], pos1[1]
		x0, y0, x1, y1 = x0*(w//WIDTH), y0*(h//HEIGHT), x1*(w//WIDTH), y1*(h//HEIGHT)
		draw.polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)], outline=outline)
	return pannel

def gav_anchor():
	coords = []
	for _,_,fs in os.walk(json_path):
		for f in fs:
			with open(json_path+f, 'r') as jfile:
				dat = json.load(jfile)
				for k in dat:
					c0, c1 = dat[k]
					w, h = c1[0]-c0[0], c1[1]-c0[1]
					coords.append((w,h))
	return (sum([e[0] for e in coords])/len(coords), sum([e[1] for e in coords])/len(coords))	# global average anchor

def output_process():
	'''
		remove all the bbox-vec with objectness-confidence lower than threshold (by set to 0-vec),
		de-duplicate the class activation (by set the dups to 0-vec)
	'''
	pass

def label_2_bbox(raw_shape, input_shape, label, anchor, num_class, downscale):
	'''
		with on global average anchor things could be much easier..
		x, y, w, h in network to x0, y0, x1, y1 associated to raw input size
	'''
	w_ratio, h_ratio = float(input_shape[0])/raw_shape[0], float(input_shape[1])/raw_shape[1]
	anchor = (anchor[0]*w_ratio, anchor[1]*h_ratio)
	jitter = 1e-7
	bbox = {}
	for x in range(input_shape[0]//downscale):
		for y in range(input_shape[1]//downscale):
			xt, yt, wt, ht = label[x,y,:4]+jitter
			if label[x,y,4] > THRESHOLD:
				offx, offy = (abs(wt)/wt)*math.e**abs(wt), (abs(ht)/ht)*math.e**abs(ht)
				wa, ha = anchor[0]+offx*2, anchor[1]+offy*2
				xc, yc = (xt+x)*downscale, (yt+y)*downscale
				x0, y0, x1, y1 = xc-wa/2, yc-ha/2, xc+wa/2, yc+ha/2
				idx = list(label[x,y,5:]).index(max(label[x,y,5:]))
				bbox[BBOX_DICT_REVERSE[idx]] = [[x0,y0],[x1,y1]]
	return bbox

def label_transfer(raw_shape, input_shape, label, anchor, downscale):
	'''
		from logged xc, yc, wt, ht to topleft and bottomright x0,y0, x1,y1
	'''
	w_ratio, h_ratio = float(input_shape[0])/raw_shape[0], float(input_shape[1])/raw_shape[1]
	anchor = (anchor[0]*w_ratio, anchor[1]*h_ratio)
	for x in range(label.shape[0]):
		for y in range(label.shape[1]):
			xt, yt, wt, ht = label[x,y,:4]
			if label[x,y,4] > 0:
				offx, offy = (abs(wt)/wt)*math.e**abs(wt), (abs(ht)/ht)*math.e**abs(ht)
				wa, ha = anchor[0]+offx*2, anchor[1]+offy*2
				xc, yc = (xt+x)*downscale, (yt+y)*downscale
				x0, y0, x1, y1 = xc-wa/2, yc-ha/2, xc+wa/2, yc+ha/2
				label[x,y,:4] = np.array([x0, y0, x1, y1])
	return label

def bbox_2_label(raw_shape, input_shape, bbox_json, anchor, num_class, downscale):
	'''
		...
	'''
	grid_shape = (input_shape[0]//downscale, input_shape[1]//downscale)
	label = np.zeros((input_shape[0]//downscale, input_shape[1]//downscale, 5+num_class))
	sum_row = np.array([0 for i in range(num_class+5)], dtype='float64')
	w_ratio, h_ratio = float(input_shape[0])/raw_shape[0], float(input_shape[1])/raw_shape[1]
	anchor = (anchor[0]*w_ratio, anchor[1]*h_ratio)
	for bk in BBOX_KEY_LIST:
		class_idx = BBOX_DICT[bk]
		bbox = bbox_json[bk]
		bbox = ((bbox[0][0]*w_ratio, bbox[0][1]*h_ratio), (bbox[1][0]*w_ratio, bbox[1][1]*h_ratio))
		cur_max_iou, cur_max_x, cur_max_y, same_count = 0, 0, 0, 0
		iou_dict = []
		for y in range(grid_shape[1]):
			for x in range(grid_shape[0]):
				x0, y0 = x*downscale+downscale/2-anchor[0]/2., y*downscale+downscale/2-anchor[1]/2.
				x1, y1 = x*downscale+downscale/2+anchor[0]/2., y*downscale+downscale/2+anchor[1]/2.
				iou = round(calculate_iou(((x0,y0),(x1,y1)), bbox), 3)
				iou_dict.append((iou,((x,y))))
				if iou>=cur_max_iou and iou>0:
					cur_max_iou = iou
					cur_max_x = x
					cur_max_y = y
		x_avg, y_avg, cnt_avg = 0, 0, 0
		for coord in iou_dict:
			if coord[0] == cur_max_iou:
				x_avg += coord[1][0]
				y_avg += coord[1][1]
				cnt_avg += 1
		cur_max_x, cur_max_y = int(x_avg/cnt_avg), int(y_avg/cnt_avg)
		# transforming
		x0, y0, x1, y1 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
		xt, yt = ((x0+x1)/2.-cur_max_x*downscale)/downscale, ((y0+y1)/2.-cur_max_y*downscale)/downscale
		offx, offy = (abs(x1-x0)-anchor[0])/2., (abs(y1-y0)-anchor[1]/2.)
		wt, ht = (offx//abs(offx))*math.log(abs(offx)), (offy//abs(offy))*math.log(abs(offy))
		cur_row = [xt, yt, wt, ht, 1]+[1 if i==class_idx else 0 for i in range(num_class)]
		label[cur_max_x, cur_max_y] =+ np.array(cur_row)
	return label

def label_flatten(input_arr, output_shape):
	ttl = 0
	for out in output_shape:
		ttl += out[0]*out[1]*out[2]
	flatten = np.zeros((ttl, output_shape[0][3]))
	head = 0
	for i, out in enumerate(output_shape):
		flatten[head:head+out[0]*out[1]*out[2]] = input_arr[i].reshape((out[0]*out[1]*out[2], out[3]))
		head += out[0]*out[1]*out[2]
	return flatten

if __name__ == '__main__':
	# with open('data/jsons/0.json', 'r') as jf:
	# 	bj = json.load(jf)
	# anchor = gav_anchor()
	# res = bbox_2_label(raw_shape=(1498,955), input_shape=(640,480), bbox_json=bj, anchor=anchor, num_class=28, downscale=16)
	# print res.shape
	# print res.reshape((40*30, 33)).sum(axis=0)
	# print '-------------------------------------------------------'
	# ret = label_transfer(raw_shape=(1498,955), input_shape=(320,240), label=res, anchor=anchor, downscale=8)
	thresh_np = np.zeros((2, 10, 4, 5))
	thresh_np[:,1,:,:] = 0.5
	print thresh_np[0]