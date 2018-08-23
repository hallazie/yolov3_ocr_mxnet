import mxnet as mx
import numpy as np
import logging
import os
import time
import random

from PIL import Image
from collections import namedtuple

logging.getLogger().setLevel(logging.DEBUG)
Batch = namedtuple('Batch', ['data'])

class dummy():
	def __init__(self):
		self.height = 64
		self.width = 64
		self.batch_size = 64
		self.num_sample = 39182
		self.data, self.label = mx.nd.zeros((self.num_sample,1,self.width,self.height)), mx.nd.zeros((self.num_sample,))
		self.ctx = mx.gpu(0)
		self.model_prefix = 'params/dummy'
		self.finetune_prefix = 'finetune/dummy_finetune'
		self.dictionary = {}
		self.diction_list = []

	def init_data(self):
		path = 'data/'
		for _,_, fs in os.walk(path):
			fs = sorted(fs)
			for i, f in enumerate(fs):
				img = Image.open(path+f).resize((self.width, self.height), resample=Image.LANCZOS).convert('L')
				if random.randint(0,10) > 5:
					img = self.attack(img)
				self.data[i,0,:,:] = self.enhance(mx.nd.array(img).transpose())
				self.label[i] = int(f.split('_')[0])
				self.dictionary[i] = f
		with open('dict', 'w') as dict_file:
			dict_file.write(str(self.dictionary))

	def get_symbol(self):
		data = mx.symbol.Variable('data')
		label = mx.symbol.Variable('softmax_label')
		c1 = mx.symbol.Convolution(data=data, num_filter=64, kernel=(3,3), pad=(1,1))
		b1 = mx.symbol.BatchNorm(data=c1)
		a1 = mx.symbol.Activation(data=b1, act_type='relu')
		c2 = mx.symbol.Convolution(data=a1, num_filter=64, kernel=(3,3), pad=(1,1))
		b2 = mx.symbol.BatchNorm(data=c2)
		a2 = mx.symbol.Activation(data=b2, act_type='relu')
		p2 = mx.symbol.Pooling(data=a2, stride=(2,2), kernel=(2,2), pool_type='max')

		c3 = mx.symbol.Convolution(data=p2, num_filter=128, kernel=(3,3), pad=(1,1))
		b3 = mx.symbol.BatchNorm(data=c3)
		a3 = mx.symbol.Activation(data=b3, act_type='relu')
		c4 = mx.symbol.Convolution(data=a3, num_filter=128, kernel=(3,3), pad=(1,1))
		b4 = mx.symbol.BatchNorm(data=c4)
		a4 = mx.symbol.Activation(data=b4, act_type='relu')
		p4 = mx.symbol.Pooling(data=a4, stride=(2,2), kernel=(2,2), pool_type='max')

		c5 = mx.symbol.Convolution(data=p4, num_filter=256, kernel=(3,3), pad=(1,1))
		b5 = mx.symbol.BatchNorm(data=c5)
		a5 = mx.symbol.Activation(data=b5, act_type='relu')
		d5 = mx.symbol.Dropout(data=a5, p=0.4)
		c6 = mx.symbol.Convolution(data=d5, num_filter=256, kernel=(3,3), pad=(1,1))
		b6 = mx.symbol.BatchNorm(data=c6)
		a6 = mx.symbol.Activation(data=b6, act_type='relu')
		p6 = mx.symbol.Pooling(data=a6, stride=(2,2), kernel=(2,2), pool_type='max')

		c7 = mx.symbol.Convolution(data=p6, num_filter=256, kernel=(3,3), pad=(1,1))
		b7 = mx.symbol.BatchNorm(data=c7)
		a7 = mx.symbol.Activation(data=b7, act_type='relu')
		d7 = mx.symbol.Dropout(data=a7, p=0.4)
		c8 = mx.symbol.Convolution(data=d7, num_filter=256, kernel=(3,3), pad=(1,1))
		b8 = mx.symbol.BatchNorm(data=c8)
		a8 = mx.symbol.Activation(data=b8, act_type='relu')
		p8 = mx.symbol.Pooling(data=a8, stride=(2,2), kernel=(2,2), pool_type='max')

		c9 = mx.symbol.Convolution(data=p8, num_filter=256, kernel=(3,3), pad=(1,1))
		b9 = mx.symbol.BatchNorm(data=c9)
		a9 = mx.symbol.Activation(data=b9, act_type='relu')
		d9 = mx.symbol.Dropout(data=a9, p=0.4)
		c10 = mx.symbol.Convolution(data=d9, num_filter=256, kernel=(3,3), pad=(1,1))
		b10 = mx.symbol.BatchNorm(data=c10)
		a10 = mx.symbol.Activation(data=b10, act_type='relu')
		p10 = mx.symbol.Pooling(data=a10, stride=(2,2), kernel=(2,2), pool_type='max')

		fl0 = mx.symbol.flatten(data=p10)
		fc1 = mx.symbol.FullyConnected(data=fl0, num_hidden=4096)
		dc1 = mx.symbol.Dropout(data=fc1, p=0.4)
		bc7 = mx.symbol.BatchNorm(data=dc1)
		ac7 = mx.symbol.Activation(data=bc7, act_type='sigmoid')
		fc8 = mx.symbol.FullyConnected(data=ac7, num_hidden=3562)
		bc8 = mx.symbol.BatchNorm(data=fc8)
		ac8 = mx.symbol.Activation(data=bc8, act_type='sigmoid')
		loss = mx.symbol.SoftmaxOutput(data=ac8, label=label)
		# return mx.symbol.Group([loss, a2])
		return loss

	def train(self):
		self.init_data()
		symbol = self.get_symbol()
		diter = mx.io.NDArrayIter(data=self.data, label=self.label, batch_size=self.batch_size, shuffle=True)
		model = mx.mod.Module(symbol=symbol, context=self.ctx, data_names=('data',), label_names=('softmax_label',))
		model.bind(data_shapes=diter.provide_data, label_shapes=diter.provide_label)
		model.init_params(initializer=mx.init.Uniform(scale=.1))
		model.fit(
			diter,
			optimizer = 'adam',
			optimizer_params = {'learning_rate':0.0005},
			eval_metric = 'acc',
			batch_end_callback = mx.callback.Speedometer(self.batch_size, 10),
			epoch_end_callback = mx.callback.do_checkpoint(self.model_prefix, 5),
			num_epoch = 200,
			)

	def finetune(self):
		self.init_data()
		symbol = self.get_symbol()
		diter = mx.io.NDArrayIter(data=self.data, label=self.label, batch_size=self.batch_size, shuffle=True)
		model = mx.mod.Module(symbol=symbol, context=self.ctx, data_names=('data',), label_names=('softmax_label',))
		model.bind(data_shapes=diter.provide_data, label_shapes=diter.provide_label)
		_, arg_params, aux_params = mx.model.load_checkpoint(self.finetune_prefix, 190)
		model.set_params(arg_params, aux_params, allow_missing=True)
		model.fit(
			diter,
			optimizer = 'adam',
			optimizer_params = {'learning_rate':0.0000005},
			eval_metric = 'acc',
			batch_end_callback = mx.callback.Speedometer(self.batch_size, 10),
			epoch_end_callback = mx.callback.do_checkpoint(self.finetune_prefix, 10),
			num_epoch = 250,
			)

	def test(self):
		self.ctx = mx.cpu(0)
		with open('char_sheet.txt', 'r') as sheet_file:
			dictionary = list(sheet_file.readline().decode('utf-8'))[1:]
		with open('result.txt', 'w') as result_file:
			symbol = self.get_symbol()
			diter = mx.io.NDArrayIter(data=self.data, label=self.label, batch_size=1)
			model = mx.mod.Module(symbol=symbol, context=self.ctx, data_names=('data',))
			model.bind(for_training=False, data_shapes=diter.provide_data)
			# _, arg_params, aux_params = mx.model.load_checkpoint(self.model_prefix, 110)
			_, arg_params, aux_params = mx.model.load_checkpoint(self.finetune_prefix, 250)
			model.set_params(arg_params, aux_params, allow_missing=True)
			for _,_,fs in os.walk('test'):
				cnt, ttl = 0, len(fs)
				fs = sorted(fs)
				for file_name in fs:
					img = self.attack(Image.open('test/'+str(file_name)).resize((self.width, self.height), resample=Image.LANCZOS).convert('L'))
					# img = Image.open('test/'+str(file_name)).resize((self.width, self.height), resample=Image.LANCZOS).convert('L')
					array = mx.nd.zeros((1,1,self.width,self.height))
					array[0,0,:] = mx.nd.array(img).transpose()
					array = self.enhance(array)
					Image.fromarray(array[0][0].asnumpy().astype('uint8').transpose()).save('test_enhanced/'+str(file_name))
					model.forward(Batch([array]))
					out = list(model.get_outputs()[0][0].asnumpy())
					out = out.index(max(out))
					# if out == int(file_name.split('.')[0].split('_')[0]):
					# 	cnt += 1
					result_file.write(dictionary[out].encode('utf-8'))
			# print 'acc: %s'%str(float(cnt)/ttl)
			print 'finished'
				
	def viz(self, file_name):
		symbol = self.get_symbol()
		diter = mx.io.NDArrayIter(data=self.data, label=self.label, batch_size=1)
		model = mx.mod.Module(symbol=symbol, context=self.ctx, data_names=('data',))
		model.bind(for_training=False, data_shapes=diter.provide_data)
		_, arg_params, aux_params = mx.model.load_checkpoint(self.model_prefix, 95)
		model.set_params(arg_params, aux_params, allow_missing=True)
		img = Image.open('data/'+str(file_name)+'.png').resize((self.width, self.height), resample=Image.LANCZOS).convert('L')
		array = mx.nd.zeros((1,1,self.width,self.height))
		array[0,0,:] = mx.nd.array(img).transpose()	
		model.forward(Batch([array]))
		out = model.get_outputs()[1][0].asnumpy()
		for i in range(64):
			tmp = out[i]
			tmp = (tmp-np.amin(tmp))*(255 / (np.amax(tmp)-np.amin(tmp)))
			feature = Image.fromarray(tmp.transpose().astype('uint8')).save('viz/%s.png'%(str(i)))	

	def enhance(self, img):
		return 254*(img>200)

	def attack(self, img):
		# rotate
		try:
			# rotate
			rot = random.randint(-20,20)
			fff = Image.new('L', img.size, (255,))
			img = np.array(img.rotate(rot))
			img = 255 - (np.array(fff.rotate(rot)) - img)
			# noise
			img = np.ma.clip((np.random.random(img.shape)*5+img),0,255)
			# random-line
			dice = random.randint(0,10)
			w,h = img.shape
			if dice==8:
				seed = random.randint(10,w-10)
				img[seed-1:seed+1,:] = 2
			elif dice==9:
				seed = random.randint(10,h-10)
				img[:,seed-1:seed+1] = 2
			return Image.fromarray(img.astype('uint8'))
		except:
			print 'sample attack failed, use original data'
			return img

if __name__ == '__main__':
	dum = dummy()
	dum.test()
