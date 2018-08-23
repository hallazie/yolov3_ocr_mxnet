#coding:utf-8

import mxnet as mx

def cfg_parse():
	blocks = []
	block = {}
	with open('cfg', 'r') as cfg_file:
		lines = cfg_file.readlines()
		lines = [e.strip() for e in lines if (len(e.strip())>0 and e.strip()[0]!='#')]
		for line in lines:
			if line[0] == '[':
				blocks.append(block)
				block = {}
				block['type'] = line[1:-1]
			else:
				k,v = line.split('=')
				block[k.strip()] = v.strip()
	blocks.append(block)
	return blocks[1:]

def conv_block(prev, **kwargs):
	conv = mx.symbol.Convolution(
		data = prev,
		num_filter = int(kwargs['num_filter']),
		kernel = eval(kwargs['kernel']),
		pad = eval(kwargs['pad'])
		)
	if bool(kwargs['bn']):
		bn = mx.symbol.BatchNorm(data=conv)
		act = mx.symbol.Activation(data=bn, act_type=kwargs['act_type'])
	else:
		act = mx.symbol.Activation(data=conv, act_type=kwargs['act_type'])
	return act



if __name__ == '__main__':
	pass