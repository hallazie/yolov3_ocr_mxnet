#coding:utf-8
import codecs
import os
import pygame
import random
import numpy as np
import PIL
import json
import skimage

from PIL import Image, ImageEnhance, ImageDraw

import gen_config

class DataGen():
	def __init__(self):
		self.pannel = np.array(Image.open('sample/fp000.png').convert('RGB'))
		self.pannel_height, self.pannel_width, _ = self.pannel.shape
		pygame.init()
		self.gauss = Image.open('pool/noise/gauss.png').convert('RGB')
		self.position = {}

	def parse_config(self):
		with open('data.cfg', 'r') as cfg_file:
			pass

	def gen_basic(self, gen_num):
		shift_range = 16
		for i in range(gen_num):
			config = gen_config.gen_single()
			self.pannel = Image.open('sample/fp000.png').convert('RGB')
			self.pannel_width, self.pannel_height = self.pannel.size
			self.pannel = np.array(self.pannel).transpose()
			self.position = {}
			shift_lvl = (int(random.randint(-shift_range,shift_range)), int(random.randint(-shift_range,shift_range)))
			# shift_lvl = (0, 0)
			for kv in config:
				self.render_line(kv, config[kv], shift_lvl)
			self.add_paper_effect()
			self.deform()
			self.background()
			self.noise()
			self.enhance_contrast()
			self.rect()
			self.save(str(i))

	def render_line(self, kv, line_dict, shift_lvl):
		#TODO:添加旋转的接口
		font = pygame.font.Font(line_dict['font'], line_dict['fontsize'])
		line = line_dict['chars'].decode('utf-8')
		cur_x, cur_y = line_dict['pos'][0], line_dict['pos'][1]
		if line_dict['rdshift']:
			cur_x += shift_lvl[0]
			cur_y += shift_lvl[0]
		x0, y0 = cur_x, cur_y
		for i, char in enumerate(line):
			rtext = 255-pygame.surfarray.array2d(font.render(char, True, (0,0,0), (255,255,255)))
			w, h = rtext.shape
			try:
				self.pannel[:,cur_x:cur_x+w,cur_y:cur_y+h] = np.minimum(self.pannel[:,cur_x:cur_x+w,cur_y:cur_y+h], np.array([rtext, rtext, rtext]))
				cur_x += w
			except:
				pass
		x1, y1 = cur_x, cur_y+h
		self.position[kv] = ((x0,y0),(x1,y1))

	def save(self, fname):
		print 'saving %s'%fname
		with open('data/'+fname+'.json', 'w') as f:
			json.dump(self.position, f)
		self.pannel.save('data/'+fname+'.png')

	def iter(self):
		pass

	# ----------- attack -----------
	def attack(self):
		self.add_side_holes()
		self.add_paper_effect()
		self.background()
		self.noise()

	def add_side_holes(self):
		num_holes = random.randint(8,15)
		radius = int((self.pannel_height/num_holes)*0.2)
		side_width = radius*random.randint(3,7)

	def add_paper_effect(self):
		mask = Image.open('pool/features/%s.png'%str(random.randint(0,5)))
		mask = mask.resize((self.pannel_width, self.pannel_height)).convert('RGBA')
		self.pannel = Image.fromarray(self.pannel.astype('uint8').transpose())
		self.pannel = Image.blend(self.pannel.convert('RGBA'), mask, 0.01*random.randint(25,40))

	def noise(self):
		self.gauss = np.random.rand(self.pannel_height, self.pannel_width, 3) * 30
		self.gauss = Image.fromarray(self.gauss.astype('uint8'))
		self.pannel = Image.blend(self.pannel.convert('RGBA'), self.gauss.convert('RGBA'), 0.01*random.randint(10,30))

	def light(self):
		pass

	def shade(self):
		pass

	def rotate(self):
		self.pannel = self.pannel.rotate(random.randint(-8,8), resample=PIL.Image.BICUBIC, expand=True)

	def background(self):
		bg = Image.open('pool/background/%s.png'%str(random.randint(0,6))).convert('RGBA')
		bg_width, bg_height = bg.size
		rd_width = abs(bg_width - self.pannel_width)
		rd_height = abs(bg_height - self.pannel_height)
		rd_x, rd_y = random.randint(0,rd_width), random.randint(0,rd_height)
		bg.paste(self.pannel, (rd_x, rd_y), mask=self.pannel)
		for k in self.position:
			bx0, by0 = self.position[k][0]
			bx1, by1 = self.position[k][1]
			self.position[k] = ((bx0+rd_x, by0+rd_y), (bx1+rd_x, by1+rd_y))
		self.pannel = bg
		self.pannel_width, self.pannel_height = self.pannel.size

	def enhance_contrast(self):
		self.pannel = ImageEnhance.Contrast(self.pannel).enhance(1+random.randint(0,30)*0.01)

	def find_coeffs(self, pa, pb):
		matrix = []
		for p1, p2 in zip(pa, pb):
			matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
			matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
		A = np.matrix(matrix, dtype=np.float)
		B = np.array(pb).reshape(8)
		res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
		return np.array(res).reshape(8)

	def deform(self):
		s = 32
		if random.randint(0,10)>5:
			x0, y0 = 0, random.randint(0,s)
			x1, y1 = self.pannel_width+random.randint(-s,0), 0
			x2, y2 = self.pannel_width, self.pannel_height+random.randint(-s,0)
			x3, y3 = 0+random.randint(0,s), self.pannel_height
		else:
			x0, y0 = random.randint(0,s), 0
			x1, y1 = self.pannel_width, 0+random.randint(0,s)
			x2, y2 = self.pannel_width+random.randint(-s,0), self.pannel_height
			x3, y3 = 0, self.pannel_height+random.randint(-s,0)
		coeff = self.find_coeffs(
			[(x0, y0),(x1, y1),(x2, y2),(x3, y3)],
			[(0,0),(self.pannel_width,0),(self.pannel_width,self.pannel_height),(0,self.pannel_height)])
		self.pannel = self.pannel.transform((self.pannel_width, self.pannel_height), Image.PERSPECTIVE, coeff, Image.BICUBIC)
		a, b, c, d, e, f, g, h = [e for e in coeff]
		for k in self.position:
			bx0, by0 = self.position[k][0]
			bx1, by1 = self.position[k][1]
			bx0, by0, bx1, by1 = float(bx0)+0.5, float(by0)+0.5, float(bx1)+0.5, float(by1)+0.5
			bx2, by2 = float(a*bx0+b*by0+c)/(g*bx0+h*by0+1), float(d*bx0+e*by0+f)/(g*bx0+h*by0+1)
			bx3, by3 = float(a*bx1+b*by1+c)/(g*bx1+h*by1+1), float(d*bx1+e*by1+f)/(g*bx1+h*by1+1)
			bx2, by2, bx3, by3 = int(bx2), int(by2), int(bx3), int(by3)
			bx4, by4 = bx0+(bx0-bx2), by0+(by0-by2)
			bx5, by5 = bx1+(bx1-bx3), by1+(by1-by3)
			self.position[k] = ((bx4,by4),(bx5,by5))

	def rect(self, outline=(224,32,128)):
		draw = ImageDraw.Draw(self.pannel)
		for kv in self.position:
			pos0, pos1 = self.position[kv]
			x0, y0, x1, y1 = pos0[0], pos0[1], pos1[0], pos1[1]
			draw.polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)], outline=outline)

if __name__ == '__main__':
	gen = DataGen()
	gen.gen_basic(5)