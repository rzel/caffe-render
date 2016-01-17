

src2 = '/scratch/xiaolonw/render_data/imgs/';

test_listfile = '/nfs.yoda/xiaolonw/gan_render/genNormalCodeReg/trainlist_temp.txt'

fid = fopen(test_listfile, 'w');


filesDir = dir(src2);
sequences = {};
for i=1:numel(filesDir)
    name = filesDir(i).name;
    if name(1) ~= '.', sequences{end+1} = filesDir(i).name; end
end


for i = 1 : numel(sequences) 
	seq = sequences{i};
	filepath = [src2 seq]; 
	fprintf('%s\n', filepath);
	imglist = dir([ filepath '/*.jpg'] ); 
	for j = 1 : 20 : numel(imglist) 
		imgname = imglist(j).name; 
		imgname = [seq '/' imgname];
		fprintf(fid, '%s\n', imgname);

	end
end


fclose(fid);
