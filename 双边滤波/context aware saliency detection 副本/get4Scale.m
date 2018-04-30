function [ img_scale_1,img_scale_2,img_scale_3,img_scale_4 ] = get4Scale( img_in )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


%********************************************************************************
%1. scale_1 1
img_scale_1=img_in;
[h,w]=size(img_scale_1(:,:,1));%记录原始图像尺寸

%----------------------------------------------------------
%2.scale_2 0.8

%以下缩放为最邻近差值
img_scale_2_g(:,:,1)=gaussianSmooth( img_scale_1(:,:,1) );%先每一层进行高斯平滑,模糊时使用原图
img_scale_2_g(:,:,2)=gaussianSmooth( img_scale_1(:,:,2) );
img_scale_2_g(:,:,3)=gaussianSmooth( img_scale_1(:,:,3) );

img_scale_2(:,:,1)=imresize(img_scale_2_g(:,:,1),[h*0.8,w*0.8]);%图像缩小为第二尺度，0.8
img_scale_2(:,:,2)=imresize(img_scale_2_g(:,:,2),[h*0.8,w*0.8]);
img_scale_2(:,:,3)=imresize(img_scale_2_g(:,:,3),[h*0.8,w*0.8]);

%----------------------------------------------------------
%3.scale_3 0.5
img_scale_3_g(:,:,1)=gaussianSmooth( img_scale_2(:,:,1) );%先每一层进行高斯平滑，模糊时使用放缩0.8的图作为原图
img_scale_3_g(:,:,2)=gaussianSmooth( img_scale_2(:,:,2) );
img_scale_3_g(:,:,3)=gaussianSmooth( img_scale_2(:,:,3) );

img_scale_3(:,:,1)=imresize(img_scale_3_g(:,:,1),[h*0.5,w*0.5]);%图像缩小为第3尺度，0.5
img_scale_3(:,:,2)=imresize(img_scale_3_g(:,:,2),[h*0.5,w*0.5]);
img_scale_3(:,:,3)=imresize(img_scale_3_g(:,:,3),[h*0.5,w*0.5]);

%----------------------------------------------------------
%4.scale_4 0.3
img_scale_4_g(:,:,1)=gaussianSmooth( img_scale_3(:,:,1) );%先每一层进行高斯平滑，模糊时使用放缩后的尺度0.5作为原图
img_scale_4_g(:,:,2)=gaussianSmooth( img_scale_3(:,:,2) );
img_scale_4_g(:,:,3)=gaussianSmooth( img_scale_3(:,:,3) );

img_scale_4(:,:,1)=imresize(img_scale_4_g(:,:,1),[h*0.3,w*0.3]);%图像缩小为第4尺度，0.3
img_scale_4(:,:,2)=imresize(img_scale_4_g(:,:,2),[h*0.3,w*0.3]);
img_scale_4(:,:,3)=imresize(img_scale_4_g(:,:,3),[h*0.3,w*0.3]);

end

