#coding:utf-8

import mxnet as mx
import numpy as np
import logging
import json
import os
import util

from config import *
from PIL import Image
from collections import namedtuple

logging.getLogger().setLevel(logging.DEBUG)

model_prefix = 'params/yolo'
ctx = mx.gpu(0)
Batch = namedtuple('Batch', ['data'])

def diter(train=False):
	data_path = 'data/imgs/'
	label_path = 'data/jsons/'
	anchor = util.gav_anchor()
	if train == True:
		for _,_, fs in os.walk(data_path):
			data = np.zeros((len(fs), 3, WIDTH, HEIGHT))
			label = np.zeros((len(fs), 33, WIDTH//DOWNSAMPLE, HEIGHT//DOWNSAMPLE))
			for i, f in enumerate(fs):
				img = Image.open(data_path+f).resize((WIDTH, HEIGHT), resample=Image.LANCZOS)
				with open(label_path+f.split('.')[0]+'.json', 'r') as lj:
					bbox = json.load(lj)
					res = util.bbox_2_label(raw_shape=img.size, input_shape=(WIDTH,HEIGHT), bbox_json=bbox, anchor=anchor, num_class=28, downscale=16)
				data[i] = np.array(img.convert('RGB')).transpose()
				label[i] = res.transpose().swapaxes(1,2)
			print 'data iter gen finished'
		return mx.io.NDArrayIter(data=data, label=label, batch_size=BATCH_SIZE, shuffle=True)
	else:
		return mx.io.NDArrayIter(data=np.zeros((2,3,WIDTH,HEIGHT)), label=np.ones((2,33,WIDTH//DOWNSAMPLE,HEIGHT//DOWNSAMPLE)), batch_size=1, shuffle=True)

def res_block(data, num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act='leaky', down=2):
	c1 = conv_block(data, num_filter, kernel, stride, pad, act)
	c2 = conv_block(c1, num_filter//down, (1,1), (1,1), (0,0), act)
	c3 = conv_block(c2, num_filter, kernel, stride, pad, act)
	return c3

def conv_block(data, num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='leaky'):
	conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, workspace=64)
	bn = mx.symbol.BatchNorm(data=conv)
	if act_type == 'leaky':
		act = mx.symbol.LeakyReLU(data=bn)
	else:
		act = mx.symbol.Activation(data=bn, act_type=act_type)
	return act

def pool_block(data, stride=(2,2), kernel=(2,2), pool_type='max'):
	return mx.symbol.Pooling(data=data, stride=stride, kernel=kernel, pool_type=pool_type)

def confidence_mask(data, threshold):
	thresh_np = np.zeros((BATCH_SIZE, 33, WIDTH//DOWNSAMPLE, HEIGHT//DOWNSAMPLE))
	thresh_np[:,4,:,:] = threshold
	thresh_in = mx.init.Constant(thresh_np.tolist())
	thresh_mt = mx.symbol.var('mask', init=thresh_in)
	mask = mx.symbol.broadcast_greater(lhs=data,rhs=thresh_mt)
	return data*mask

def net():
	# 640*480
	data = mx.symbol.Variable('data')
	label = mx.symbol.Variable('softmax_label')
	c1 = conv_block(data, 32)
	# p1 = pool_block(c1)
	c2 = conv_block(c1, 64)
	p2 = pool_block(c2)				# 320
	r3 = res_block(p2, 128)
	r4 = res_block(r3, 128)
	p4 = pool_block(r4)			# 160
	r5 = res_block(p4, 256)
	r6 = res_block(r5, 256)
	r7 = res_block(r6, 256)
	p7 = pool_block(r7)			# 80
	r8 = res_block(p7, 512)
	r9 = res_block(r8, 512)
	p9 = pool_block(r9)			# 40, scale2
	r10 = res_block(p9, 512)
	r11 = res_block(r10, 512)
	c12 = conv_block(r11, num_filter=33, kernel=(1,1), stride=(1,1), pad=(0,0), act_type='relu')
	msk = confidence_mask(c12, 0.5)
	# loss = mx.symbol.LinearRegressionOutput(data=c12, label=label)
	out_xy = mx.symbol.slice(data=c12, begin=(None,0,None,None), end=(None,2,None,None))
	out_wh = mx.symbol.slice(data=c12, begin=(None,2,None,None), end=(None,4,None,None))
	out_oc = mx.symbol.slice(data=c12, begin=(None,4,None,None), end=(None,5,None,None))
	out_cs = mx.symbol.slice(data=c12, begin=(None,5,None,None), end=(None,None,None,None))
	lbl_xy = mx.symbol.slice(data=label, begin=(None,0,None,None), end=(None,2,None,None))
	lbl_wh = mx.symbol.slice(data=label, begin=(None,2,None,None), end=(None,4,None,None))
	lbl_oc = mx.symbol.slice(data=label, begin=(None,4,None,None), end=(None,5,None,None))
	lbl_cs = mx.symbol.slice(data=label, begin=(None,5,None,None), end=(None,None,None,None))
	xy_loss = mx.symbol.LogisticRegressionOutput(data=out_xy, label=lbl_xy)
	wh_loss = mx.symbol.LinearRegressionOutput(data=out_wh, label=lbl_wh)
	oc_loss = mx.symbol.LogisticRegressionOutput(data=out_oc, label=lbl_oc)
	cs_loss = mx.symbol.LogisticRegressionOutput(data=out_cs, label=lbl_cs)
	loss = mx.symbol.concat(xy_loss, wh_loss, oc_loss, cs_loss)
	return loss

def train():
	symbol = net()
	dataiter = diter(train=True)
	model = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',), label_names=('softmax_label',))
	model.bind(data_shapes=dataiter.provide_data, label_shapes=dataiter.provide_label)
	model.init_params(initializer=mx.init.Uniform(scale=.1))
	model.fit(
		dataiter,
		optimizer = 'adam',
		optimizer_params = {'learning_rate':0.0005},
		eval_metric = 'mse',
		batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 1),
		epoch_end_callback = mx.callback.do_checkpoint(model_prefix, 50),
		num_epoch = 100,
	)

def predict():
	symbol = net()
	anchor = util.gav_anchor()
	dataiter = diter(train=False)
	model = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',), label_names=('softmax_label',))
	model.bind(for_training=False, data_shapes=dataiter.provide_data)
	_, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 350)
	model.set_params(arg_params, aux_params, allow_missing=True)
	raw = Image.open('data/imgs/0.png').convert('RGB')
	w, h = raw.size
	img = np.array(raw.resize((WIDTH, HEIGHT), resample=Image.BICUBIC)).transpose().reshape((1,3,WIDTH,HEIGHT))
	model.forward(Batch([mx.nd.array(img)]))
	out = model.get_outputs()[0][0].asnumpy()
	t = out.reshape((20*15),33)
	for e in t:
		if e[4]>0:
			print [round(a,4) for a in e]
	bbox = util.label_2_bbox(raw_shape=(w,h), input_shape=(WIDTH,HEIGHT), label=out, anchor=anchor, num_class=28, downscale=DOWNSAMPLE)
	for k in bbox:
		print '%s,%s'%(k,bbox[k])
	util.rect(raw, bbox)
	print 'predict finished'
	raw.resize((w,h)).save('data/0.png')

if __name__ == '__main__':
	train()