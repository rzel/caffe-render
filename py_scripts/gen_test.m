jpgfoldr = '/nfs/hn46/xiaolonw/render_cnncode/results/';

test_listfile = '/nfs.yoda/xiaolonw/gan_render/genNormalCodeReg/testlist_temp.txt'

fid = fopen(test_listfile, 'w');

for i = 1 : 300
	s = sprintf('%04d.jpg', i );
	fprintf(fid, '%s\n', s);

end

fclose(fid);


