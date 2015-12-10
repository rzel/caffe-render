import numpy as np
import cv2
# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

test_listfile = '/nfs.yoda/xiaolonw/gan_render/genNormalCodeReg/trainlist.txt'
result_folder = '/nfs.yoda/xiaolonw/gan_render/results/'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('/nfs/hn46/xiaolonw/render_cnncode/caffe-render/render_scripts/deconv_small/test.prototxt',
                '/nfs.yoda/xiaolonw/gan_render/models/deconv_sgd/train_iter_30000.caffemodel',
                caffe.TEST)


#'/nfs.yoda/xiaolonw/gan_render/models/deconv_sgd_gaussian2/train_iter_5000.caffemodel',
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

		lbl_set = lbl.split('/')
		imgname = result_folder + '/' + lbl_set[-2] + '_' + lbl_set[-1]
		
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



