import numpy as np
import cv2
# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
import struct
from subprocess import call

sys.path.insert(0, caffe_root + 'python')

import caffe

test_listfile = '/nfs.yoda/xiaolonw/gan_render/genNormalCodeReg/trainlist_rand.txt'
result_folder = '/nfs.yoda/xiaolonw/gan_render/results/'
source_folder = '/nfs.yoda/xiaolonw/gan_render/data/'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('test.prototxt',
                '/nfs.yoda/xiaolonw/gan_render/models/deconv_sgd_bnadam4/train_iter_20000.caffemodel',
                caffe.TEST)


test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
data_counts = len(test_list)
batch_size = net.blobs['data'].data.shape[0]


batch_count = int(np.ceil(data_counts * 1.0 / batch_size))
accuracy = 0

height = 72
width = 72
chan  = 3

for i in range(1):
	out = net.forward()
	# transformer = caffe.io.Transformer({'conv9': out['conv9'].shape})
	# transformer.set_transpose('conv9', (1,2,0))
	for j in range(batch_size):
		id = j
		if id >= data_counts:
			break
		lbl = test_list[id].split(' ')[1]
		fname = test_list[id].split(' ')[0]

		source_lbl = source_folder + lbl
		source_fname = source_folder + fname

		lbl_set = lbl.split('/')
		imgname = result_folder + '/' + lbl_set[-2] + '_' + lbl_set[-1]

		target_lbl = imgname + '_norm.jpg'		
		target_fname = imgname + '_ori.jpg'

		# cmd = 'cp ' + source_lbl + ' ' + target_lbl
		# call(cmd) 
		imori = cv2.imread(source_lbl)
		cv2.imwrite(target_lbl, imori)

		f = open(source_fname, 'rb');
		normi = np.zeros((height, width, chan))
		for c in range(chan):
			for h in range(height):
				for w in range(width):
					(x, ) =  struct.unpack('f', f.read(4))
					normi[h,w,2 - c] = x
		f.close()
		normi = np.uint8(normi)
		cv2.imwrite(target_fname, normi)		

		
		timg = out['conv9'][j]
		img = np.zeros((height, width, chan)) 
		img[:,:,0] = timg[2,:]
		img[:,:,1] = timg[1,:]
		img[:,:,2] = timg[0,:]
		img = img * 127 + 104
		img = np.uint8(img)

		# imgname = result_folder + lbl
		print imgname

		cv2.imwrite(imgname, img)



