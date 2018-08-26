#coding:utf-8

import mxnet as mx

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

def net():
	# 640*480
	data = mx.symbol.Variable('data')
	c1 = conv_block(data, 32)
	p1 = pool_block(c1)			#320
	c2 = conv_block(p1, 64)
	p2 = pool_block(c2)			# 160
	r3 = res_block(p2, 128)
	r4 = res_block(r3, 128)
	p4 = pool_block(r4+p2)			# 80
	r5 = res_block(p4, 192)
	r6 = res_block(p5, 192)
	r7 = res_block(p6, 192)
	p7 = pool_block(r7+p4)			# 40, scale1
	r8 = res_block(p7, 256)
	r9 = res_block(r8, 253)
	p9 = pool_block(r9+p7)			# 20, scale2
	r10 = res_block(p9, 384)
	r11 = res_block(r10, 384)
	p11 = pool_block(r11+p9)		# 10, scale3
	