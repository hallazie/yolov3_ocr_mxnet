#coding:utf-8

from PIL import Image, ImageDraw, ImageFont
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
	font = ImageFont.truetype('font/mriamc.ttf',16)
	for kv in bbox:
		pos0, pos1 = bbox[kv]
		x0, y0, x1, y1 = pos0[0], pos0[1], pos1[0], pos1[1]
		# x0, y0, x1, y1 = x0*(w//WIDTH), y0*(h//HEIGHT), x1*(w//WIDTH), y1*(h//HEIGHT)
		draw.polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)], outline=outline)
		draw.text((x0,y0-16),kv,font=font,fill=outline)
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

def label_2_bbox(raw_shape, input_shape, label, anchor, num_class, downscale, threshold=0.5):
	'''
		with on global average anchor things could be much easier..
		x, y, w, h in network to x0, y0, x1, y1 associated to raw input size
	'''
	w_ratio, h_ratio = float(input_shape[0])/raw_shape[0], float(input_shape[1])/raw_shape[1]
	anchor = (anchor[0]*w_ratio, anchor[1]*h_ratio)
	jitter = 1e-10
	bbox = {}
	label = label.reshape((input_shape[0]//downscale, input_shape[1]//downscale, 33))
	for x in range(input_shape[0]//downscale):
		for y in range(input_shape[1]//downscale):
			xt, yt, wt, ht = label[x,y,:4]+jitter
			if label[x,y,5:].sum() > threshold:
				offx, offy = (abs(wt)//wt)*math.e**abs(wt), (abs(ht)//ht)*math.e**abs(ht)
				wa, ha = anchor[0]+offx*2, anchor[1]+offy*2
				xc, yc = (xt+x)*downscale, (yt+y)*downscale
				x0, y0, x1, y1 = xc-wa/2, yc-ha/2, xc+wa/2, yc+ha/2
				idx = list(label[x,y,5:]).index(max(label[x,y,5:]))
				bbox[BBOX_DICT_REVERSE[idx]] = [[x0/w_ratio,y0/h_ratio],[x1/w_ratio,y1/h_ratio]]
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

		## calculate IOU and get x, y coords, seems not neccessary for our case.
		# cur_max_iou, cur_max_x, cur_max_y, same_count = 0, 0, 0, 0
		# iou_dict = []
		# for y in range(grid_shape[1]):
		# 	for x in range(grid_shape[0]):
		# 		x0, y0 = x*downscale+downscale/2-anchor[0]/2., y*downscale+downscale/2-anchor[1]/2.
		# 		x1, y1 = x*downscale+downscale/2+anchor[0]/2., y*downscale+downscale/2+anchor[1]/2.
		# 		iou = round(calculate_iou(((x0,y0),(x1,y1)), bbox), 3)
		# 		iou_dict.append((iou,((x,y))))
		# 		if iou>=cur_max_iou and iou>0:
		# 			cur_max_iou = iou
		# 			cur_max_x = x
		# 			cur_max_y = y
		# x_avg, y_avg, cnt_avg = 0, 0, 0
		# for coord in iou_dict:
		# 	if coord[0] == cur_max_iou:
		# 		x_avg += coord[1][0]
		# 		y_avg += coord[1][1]
		# 		cnt_avg += 1
		# cur_max_x, cur_max_y = int(x_avg/cnt_avg), int(y_avg/cnt_avg)
		# x0, y0, x1, y1 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
		# xt, yt = ((x0+x1)/2.-cur_max_x*downscale)/downscale, ((y0+y1)/2.-cur_max_y*downscale)/downscale
		# offx, offy = (abs(x1-x0)-anchor[0])/2., (abs(y1-y0)-anchor[1]/2.)
		# wt, ht = (offx//abs(offx))*math.log(abs(offx)), (offy//abs(offy))*math.log(abs(offy))
		# cur_row = [xt, yt, wt, ht, 1]+[1 if i==class_idx else 0 for i in range(num_class)]
		# label[cur_max_x, cur_max_y] =+ np.array(cur_row)

		## transforming
		x0, y0, x1, y1 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
		xc, yc = (x0+x1)/2., (y0+y1)/2. 
		xb, yb = (xc-(xc//downscale)*downscale)/float(downscale), (yc-(yc//downscale)*downscale)/float(downscale)
		wc, hc = (abs(x1-x0)-anchor[0])/2., (abs(y1-y0)-anchor[1])/2.
		wb, hb = (wc//abs(wc))*math.log(abs(wc)), (hc//abs(hc))*math.log(abs(hc))
		cur_row = [xb, yb, wb, hb, 1]+[1 if i==class_idx else 0 for i in range(num_class)]
		label[xc//downscale, yc//downscale] =+ np.array(cur_row)
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

def diter(train=False):
	data_path = 'data/imgs/'
	label_path = 'data/jsons/'
	anchor = gav_anchor()
	if train == True:
		for _,_, fs in os.walk(data_path):
			fs = sorted(fs)[:10]
			data = np.zeros((len(fs), 3, WIDTH, HEIGHT))
			label = np.zeros((len(fs), 33, WIDTH//DOWNSAMPLE, HEIGHT//DOWNSAMPLE))
			for i, f in enumerate(fs):
				img = Image.open(data_path+f)
				raw_size = img.size
				img = img.resize((WIDTH, HEIGHT), resample=Image.BICUBIC)
				with open(label_path+f.split('.')[0]+'.json', 'r') as lj:
					bbox = json.load(lj)
					res = bbox_2_label(raw_shape=raw_size, input_shape=(WIDTH,HEIGHT), bbox_json=bbox, anchor=anchor, num_class=28, downscale=DOWNSAMPLE)
				res = res.reshape(((WIDTH//DOWNSAMPLE)*(HEIGHT//DOWNSAMPLE), 33))
				for line in res:
					print [round(e,3) for e in line]
				print '------------------------------------------'
			print 'data iter gen finished'

if __name__ == '__main__':
	# with open('data/jsons/0.json', 'r') as jf:
	# 	bj = json.load(jf)
	# anchor = gav_anchor()
	# res = bbox_2_label(raw_shape=(1498,955), input_shape=(640,480), bbox_json=bj, anchor=anchor, num_class=28, downscale=16)
	# print res.shape
	# res = res.reshape((40*30, 33))
	# for e in res:
	# 	if e[4]>0:
	# 		print [round(k,3) for k in e]
	# print '-------------------------------------------------------'
	# ret = label_2_bbox(raw_shape=(1498,955), input_shape=(640,480), label=res, anchor=anchor, num_class=28, downscale=16)
	# print ret
	diter(True)