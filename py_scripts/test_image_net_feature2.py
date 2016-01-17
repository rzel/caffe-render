import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe


jpgfoldr = '/scratch/xiaolonw/render_data/imgs/'
test_listfile = '/nfs.yoda/xiaolonw/gan_render/genNormalCodeReg/trainlist_temp.txt'
# train_listfile = '/nfs.yoda/xiaolonw/gan_render/genNormalCodeReg/trainlist_rand.txt'
result_file = '/nfs.yoda/xiaolonw/gan_render/trainfc_results.txt'
# train_result_file = 'train_results.txt' 

caffe.set_device(2)
caffe.set_mode_gpu()
net = caffe.Net('/nfs.yoda/xiaolonw/gan_render/PLACES/places205CNN_deploy_FC7.prototxt',
                '/nfs.yoda/xiaolonw/gan_render/PLACES/places205CNN_iter_300000.caffemodel',
                caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # Transformer is pycaffe object 
transformer.set_transpose('data', (2,0,1))  # put the channel in the first place 
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel value 3 * 1 vector 
transformer.set_raw_scale('data', 255)  # python image is in the scale of [0,1], rescale to [0,255] 
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB, caused by : OPENCV used BGR. 


test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
data_counts = len(test_list)
batch_size = net.blobs['data'].data.shape[0]
batch_count = int(np.ceil(data_counts * 1.0 / batch_size))

print(batch_size)
print(batch_count)

accuracy = 0
f = open(result_file, 'w')
for i in range(batch_count):
	fnames = []
	cnt = 0
	for j in range(batch_size):
		id = i * batch_size + j
		if id >= data_counts: 
			break
		cnt = cnt + 1

		fname = test_list[id]
		fname = jpgfoldr + fname 
		if j == 1 :
			print(fname)

		fnames.append(fname) 
		net.blobs['data'].data[...][j] = transformer.preprocess('data', caffe.io.load_image(fname))

	out = net.forward()

	for j in range(cnt):
		props = out['fc7'][j]
		f.write(fnames[j])
		for k in range(len(props)):
			f.write(' {0: .6f}'.format(props[k]))
		f.write('\n')

f.close()


