import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

test_listfile = '/home/xiaolonw/video_dataset/hmdb_frames_org_scripts/testimglist_full_imgs_2.txt'
result_file = 'cls_results.txt'
pair_size = 25

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('/nfs/hn46/xiaolonw/action_code/caffe-video_siamese/siamese_scripts_vgg_hmdb/hmdb_vgg16_app_cls_4/imagenet_test.prototxt',
                '/nfs.yoda/xiaolonw/vol1_video_models/hmdb_vgg16_app_cls_4/video__iter_60000.caffemodel',
                caffe.TEST)


test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
data_counts = len(test_list)
batch_size = net.blobs['data'].data.shape[0]
video_num = batch_size / pair_size

batch_count = int(np.ceil(data_counts * 1.0 / batch_size))
accuracy = 0
f = open(result_file, 'w')
for i in range(batch_count):
	out = net.forward()
	for j in range(video_num):
		props = 0
		id = (i * video_num + j) * pair_size
		if id >= data_counts:
			break
		lbl = int(test_list[id].split(' ')[1])
		fname = test_list[id].split(' ')[0]
		
		print id / pair_size, fname

		for k in range(pair_size):
			prop = out['softmax'][j * pair_size + k]
			props = props + prop
		pred_lbl = props.argmax()
		if pred_lbl == lbl:
			accuracy = accuracy + 1
		
		f.write(fname)
		f.write('{0: d}'.format(lbl))
		for k in range(len(props)):
			f.write('{0: .7f}'.format(props[k]))
		f.write('\n')

f.close()

accuracy = accuracy * 1.0 / ( data_counts / pair_size ) 

print accuracy


