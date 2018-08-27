#coding:utf-8

import mxnet as mx
import numpy as np
import dataiter

from config import *

model_prefix = 'params/yolo'
ctx = mx.gpu(0)

def res_block(data, num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act='leaky', down=2):
	c1 = conv_block(data, num_filter, kernel, stride, pad, act)
	c2 = conv_block(c1, num_filter//down, (1,1), (1,1), 0, act)
	c3 = conv_block(c2, num_filter, kernel, stride, pad, act)
	return c3

def conv_block(data, num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='leaky'):
	conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
	bn = mx.symbol.BatchNorm(data=conv)
	if act_type == 'leaky':
		act = mx.symbol.LeakyReLU(data=bn)
	else:
		act = mx.symbol.Activation(data=bn, act_type=act_type)
	return act

def pool_block(data, stride=(2,2), kernel=(2,2), pool_type='max'):
	return mx.symbol.Pooling(data=data, stride=stride, kernel=kernel, pool_type=pool_type)

def confidence_mask(data, threshold):
	# mask = data[:,:,4]>threshold
	# return data*mask
	return data

def net():
	# 640*480
	data = mx.symbol.Variable('data')
	label = mx.symbol.Variable('label')
	c1 = conv_block(data, 32)
	# p1 = pool_block(c1)
	c2 = conv_block(c1, 64)
	p2 = pool_block(c2)				# 320
	r3 = res_block(p2, 128)
	r4 = res_block(r3, 128)
	p4 = pool_block(r4)			# 160
	r5 = res_block(p4, 192)
	r6 = res_block(r5, 192)
	r7 = res_block(r6, 192)
	p7 = pool_block(r7)			# 80
	r8 = res_block(p7, 256)
	r9 = res_block(r8, 253)
	p9 = pool_block(r9)			# 40, scale2
	r10 = res_block(p9, 384)
	r11 = res_block(r10, 384)
	msk = confidence_mask(conv_block(r11, num_filter=33, kernel=(1,1), stride=(1,1), pad=(0,0), act_type='relu'), 0.8)
	return mx.symbol.LinearRegressionOutput(data=msk, label=label)
	
def train():
	symbol = net()
	# diter = dataiter.diter()
	diter = mx.io.NDArrayIter(data=np.zeros((1,WIDTH,HEIGHT,3)), label=np.zeros((1,WIDTH//DOWNSAMPLE,HEIGHT//DOWNSAMPLE,33)), batch_size=1, shuffle=True)
	model = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',), label_names=('label',))
	model.bind(data_shapes=diter.provide_data, label_shapes=diter.provide_label)
	model.init_params(initializer=mx.init.Uniform(scale=.1))
	model.fit(
		diter,
		optimizer = 'adam',
		optimizer_params = {'learning_rate':0.0005},
		eval_metric = 'mse',
		batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 1),
		epoch_end_callback = mx.callback.do_checkpoint(model_prefix, 1),
		num_epoch = 200,
		)

if __name__ == '__main__':
	train()