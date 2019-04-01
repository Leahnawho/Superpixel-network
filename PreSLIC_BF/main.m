
ObjDir = 'E:\YYN' %path
tnum = 450;%number of images

for i = 1:1:tnum  
    m=i-1;
	bgFile = [ObjDir,'\',int2str(m),'.tif']; 
	imgs=imread(bgFile);

% simple linear iterativeclustering;
	[l, Am, C] = slic(imgs,200, 10, 1, 'median');

%show the superpixels results
	figure(i);
	I1=drawregionboundaries(l, imgs, [0 0 0]);
	imshow(I1);
	hold on

	paths=ObjDir; 
% Write the picture of the i-th frame as 'super_i.bmp' and save it in a folder.
	imwrite(I1,[paths,'\superpixels\super_',strcat(int2str(m),'.tif')]); 

end

%Bilateral filter function
for j = 1:1:tnum  
    k=j-1;
    bgFile = [ObjDir,'\superpixels\super_',int2str(k),'.tif'];
    imgs=imread(bgFile);
    imgs=double(imgs)/255;

    w    = 5;      % bilateral filter half-width
    sigma = [4 0.1]; % bilateral filter standard deviations:sigma_d and sigma_r

    I2=bfilter2(imgs,w,sigma);
    %Show the results.
    figure(k+1)
    imshow(I2)
    hold on
    imwrite(I2,[paths,'\BF\',strcat(int2str(k),'_train.tif')]); 
end

