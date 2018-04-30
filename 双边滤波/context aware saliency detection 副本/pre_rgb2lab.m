function [ img_lab ] = pre_rgb2lab( img_in,max_side )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%img_rgb:缩放后的rgb图
%img_lab：rgb2lab后的图
%h，w，图像高宽



img=double(img_in);


[h_t,w_t]=size(img(:,:,1));%得到原图的长宽

if w_t>h_t              %看长和宽哪个比较大,将原图放缩至大值为250
  
    img_rgb(:,:,1)=imresize(img(:,:,1),[(max_side/w_t)*h_t,max_side],'bilinear');%用双线性差值法
    img_rgb(:,:,2)=imresize(img(:,:,2),[(max_side/w_t)*h_t,max_side],'bilinear');
    img_rgb(:,:,3)=imresize(img(:,:,3),[(max_side/w_t)*h_t,max_side],'bilinear');
else
     img_rgb(:,:,1)=imresize(img(:,:,1),[max_side,(max_side/h_t)*w_t],'bilinear');
     img_rgb(:,:,2)=imresize(img(:,:,2),[max_side,(max_side/h_t)*w_t],'bilinear');
     img_rgb(:,:,3)=imresize(img(:,:,3),[max_side,(max_side/h_t)*w_t],'bilinear');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
   
%========================================================   

%=====================RGB2Lab=============================
img_lab_o=RGB2Lab(img_rgb);

img_lab(:,:,1)=img_lab_o(:,:,1)*255/100;      %L通道从0到100转到0到255
img_lab(:,:,2)=(img_lab_o(:,:,2)+120)*255/200;%a，通道从-120到120转到0到255
img_lab(:,:,3)=(img_lab_o(:,:,3)+120)*255/200;

%========================================================
%归一化，从0到255归到0到1
img_lab=img_lab/255;



end

