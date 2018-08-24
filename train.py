#coding:utf-8

# import data_gen

if __name__ == '__main__':
	# gen = data_gen.DataGen()
	# gen.gen_basic(20)
	h = 480
	w = 640
	num_anchors = 9
	num_classes = 80
	print [(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes+5) for l in range(3)]