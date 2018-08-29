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

cur_thresh = 0.1

def diter(train=False):
	data_path = 'data/imgs/'
	label_path = 'data/jsons/'
	anchor = util.gav_anchor()
	if train == True:
		for _,_, fs in os.walk(data_path):
			fs = sorted(fs)[:2]
			data = np.zeros((len(fs), 3, WIDTH, HEIGHT))
			label = np.zeros((len(fs), 33, WIDTH//DOWNSAMPLE, HEIGHT//DOWNSAMPLE))
			for i, f in enumerate(fs):
				img = Image.open(data_path+f)
				raw_size = img.size
				img = img.resize((WIDTH, HEIGHT), resample=Image.BICUBIC)
				with open(label_path+f.split('.')[0]+'.json', 'r') as lj:
					bbox = json.load(lj)
					res = util.bbox_2_label(raw_shape=raw_size, input_shape=(WIDTH,HEIGHT), bbox_json=bbox, anchor=anchor, num_class=28, downscale=DOWNSAMPLE)
				data[i] = np.array(img.convert('RGB')).transpose()
				label[i] = res.transpose().swapaxes(1,2)
			print 'data iter gen finished'
		return mx.io.NDArrayIter(data=data, label=label, batch_size=BATCH_SIZE, shuffle=True)
	else:
		return mx.io.NDArrayIter(data=np.zeros((1,3,WIDTH,HEIGHT)), batch_size=1)

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
	elif act_type == 'none':
		act = bn
	else:
		act = mx.symbol.Activation(data=bn, act_type=act_type)
	return act

def pool_block(data, stride=(2,2), kernel=(2,2), pool_type='max'):
	return mx.symbol.Pooling(data=data, stride=stride, kernel=kernel, pool_type=pool_type)

def confidence_mask_top1_at_each_class(data, threshold, train):
	# TODO: change to top1 on each class.
	bsize = BATCH_SIZE if train else 1
	obj_vec = mx.symbol.slice(data, begin=(None,5,None,None), end=(None,None,None,None))
	obk_vec = mx.symbol.reshape(obj_vec, shape=(bsize, 28, (WIDTH//DOWNSAMPLE)*(HEIGHT//DOWNSAMPLE)))
	top_vec = mx.symbol.topk(obk_vec, axis=2, k=1, ret_typ='mask')
	rsp_vec = mx.symbol.sum_axis(top_vec, axis=1)
	tpp_vec = mx.symbol.reshape(rsp_vec, shape=(bsize,1,WIDTH//DOWNSAMPLE,HEIGHT//DOWNSAMPLE))
	clp_vec = mx.symbol.clip(tpp_vec, 0, 1)
	blk_vec = mx.symbol.BlockGrad(clp_vec)
	ret_vec = mx.symbol.broadcast_mul(lhs=data, rhs=blk_vec)
	return ret_vec, blk_vec

def confidence_mask_thresh(data, threshold, train):
	bsize = BATCH_SIZE if train else 1
	obj_vec = mx.symbol.slice(data, begin=(None,4,None,None), end=(None,5,None,None))
	ths_vec = mx.init.Constant((np.ones((bsize,1,WIDTH//DOWNSAMPLE,HEIGHT//DOWNSAMPLE))*threshold).tolist())
	tht_vec = mx.sym.Variable('mask', shape=(bsize,1,WIDTH//DOWNSAMPLE,HEIGHT//DOWNSAMPLE), init=ths_vec)
	rsp_vec = mx.symbol.broadcast_greater(lhs=obj_vec, rhs=tht_vec)
	blk_vec = mx.symbol.BlockGrad(rsp_vec)
	return mx.symbol.broadcast_mul(lhs=data, rhs=blk_vec), tht_vec

def diverse_act(data):
	xy_part = mx.symbol.slice(data, begin=(None,0,None,None), end=(None,2,None,None))
	wh_part = mx.symbol.slice(data, begin=(None,2,None,None), end=(None,4,None,None))
	rs_part = mx.symbol.slice(data, begin=(None,4,None,None), end=(None,None,None,None))
	xy_act = mx.symbol.Activation(xy_part, act_type='sigmoid')
	rs_act = mx.symbol.Activation(rs_part, act_type='sigmoid')
	return mx.symbol.concat(xy_act, wh_part, rs_act)

def net(train):
	# 640*480
	data = mx.symbol.Variable('data')
	label = mx.symbol.Variable('softmax_label')
	# c1 = conv_block(data, 32)
	# # p1 = pool_block(c1)				
	# c2 = conv_block(c1, 64)
	# p2 = pool_block(c2)
	# r3 = res_block(p2, 128)
	# r4 = res_block(r3, 128)
	# p4 = pool_block(r4)
	# r5 = res_block(p4, 256)
	# r6 = res_block(r5, 256)
	# r7 = res_block(r6, 256)
	# d7 = mx.symbol.Dropout(r7, p=0.2)
	# p7 = pool_block(d7)
	# r8 = res_block(p7, 512)
	# r9 = res_block(r8, 512)
	c1 = conv_block(data, 32)
	c2 = conv_block(c1, 64)
	p2 = pool_block(c2)
	c3 = conv_block(p2, 128)
	c4 = conv_block(c3, 128)
	p4 = pool_block(c4)
	c5 = conv_block(p4, 256)
	c6 = conv_block(c5, 256)
	c7 = conv_block(c6, 256)
	p7 = pool_block(c7)
	c8 = conv_block(p7, 512)
	c9 = conv_block(c8, 512)
	# d9 = mx.symbol.Dropout(r9, p=0.2)
	c12 = conv_block(c9, num_filter=33, kernel=(1,1), stride=(1,1), pad=(0,0), act_type='none')
	c13 = diverse_act(c12)
	msk, rsp_vec = confidence_mask_thresh(c13, THRESHOLD, train)

	if not train:
		return mx.symbol.Group([msk, rsp_vec])

	# loss = mx.symbol.LinearRegressionOutput(data=msk, label=label)
	out_xy = mx.symbol.slice(data=msk, begin=(None,0,None,None), end=(None,2,None,None))
	out_wh = mx.symbol.slice(data=msk, begin=(None,2,None,None), end=(None,4,None,None))
	out_oc = mx.symbol.slice(data=msk, begin=(None,4,None,None), end=(None,5,None,None))
	out_cs = mx.symbol.slice(data=msk, begin=(None,5,None,None), end=(None,None,None,None))
	lbl_xy = mx.symbol.slice(data=label, begin=(None,0,None,None), end=(None,2,None,None))
	lbl_wh = mx.symbol.slice(data=label, begin=(None,2,None,None), end=(None,4,None,None))
	lbl_oc = mx.symbol.slice(data=label, begin=(None,4,None,None), end=(None,5,None,None))
	lbl_cs = mx.symbol.slice(data=label, begin=(None,5,None,None), end=(None,None,None,None))
	xy_loss = mx.symbol.LogisticRegressionOutput(data=out_xy, label=lbl_xy)
	wh_loss = mx.symbol.LinearRegressionOutput(data=out_wh, label=lbl_wh)
	oc_loss = mx.symbol.LogisticRegressionOutput(data=out_oc, label=lbl_oc)
	cs_loss = mx.symbol.SoftmaxOutput(data=out_cs, label=lbl_cs)
	loss = mx.symbol.concat(xy_loss, wh_loss, oc_loss, cs_loss)
	return loss

def train():
	symbol = net(train=True)
	dataiter = diter(train=True)
	model = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',), label_names=('softmax_label',))
	model.bind(data_shapes=dataiter.provide_data, label_shapes=dataiter.provide_label)
	model.init_params(initializer=mx.init.Uniform(scale=.1))
	model.fit(
		dataiter,
		optimizer = 'adadelta',
		# optimizer_params = {'learning_rate':0.0005},
		optimizer_params = {'rho':0.9, 'epsilon':1e-7},
		eval_metric = 'loss',
		batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 5),
		epoch_end_callback = mx.callback.do_checkpoint(model_prefix, 200),
		num_epoch = 2000,
	)

def predict():
	symbol = net(train=False)
	anchor = util.gav_anchor()
	dataiter = diter(train=False)
	model = mx.mod.Module(symbol=symbol, context=mx.cpu(0), data_names=('data',), label_names=('softmax_label',))
	model.bind(for_training=False, data_shapes=dataiter.provide_data)
	_, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 600)
	arg_params['mask'] = mx.nd.ones((1,1,WIDTH//DOWNSAMPLE,HEIGHT//DOWNSAMPLE))*THRESHOLD
	model.set_params(arg_params, aux_params, allow_missing=True)
	raw = Image.open('data/imgs/0.png').convert('RGB')
	w, h = raw.size
	img = np.array(raw.resize((WIDTH, HEIGHT), resample=Image.BICUBIC)).transpose().reshape((1,3,WIDTH,HEIGHT))
	model.forward(Batch([mx.nd.array(img)]))

	# blk = model.get_outputs()[1][0].asnumpy()
	# for line in blk[0]:
	# 	print [round(e,3) for e in line]

	out = model.get_outputs()[0][0].asnumpy().transpose().swapaxes(1,0)
	for i in range(30):
		for j in range(22):
			if True:
				print '%s,%s--->%s'%(i,j,[round(e,8) for e in out[i,j,:8]])
	bbox = util.label_2_bbox(raw_shape=(w,h), input_shape=(WIDTH,HEIGHT), label=out, anchor=anchor, num_class=28, downscale=DOWNSAMPLE, threshold=0.1)
	for k in bbox:
		print '%s,%s'%(k,bbox[k])
	print 'predict finished'
	raw = util.rect(raw, bbox, outline=(32,128,224))
	raw.save('data/0.png')

if __name__ == '__main__':
	predict()