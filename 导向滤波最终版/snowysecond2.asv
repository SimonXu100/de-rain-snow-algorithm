clc
clear
close all;
%%
%作者： 徐书森  
%完成时间：2016年7月21日
%功能描述：利用雨雪光学统计性特征，实现基于导向滤波的内容自适应的去雨雪算法
%本程序中以去雪为例
%%
%读入要处理的图，并进行分通道，分别得到三个通道的图像
I= imread('snowybox_small.jpg');
Itemp = double(I)/255;
I_R = Itemp(:,:,1);
I_G = Itemp(:,:,2);
I_B = Itemp(:,:,3);



w     = 5;       % 滤波器半宽
sigma = [3 0.1]; % 导向滤波的两个参数，前边为sigma—d, sigma-r

%对三个通道分别进行双边滤波
I_R = I_R+0.03*randn(size(I_R));
I_R(I_R<0) = 0; I_R(I_R>1) = 1;
Ibf_R = bfilter2(I_R,w,sigma);


I_G = I_G+0.03*randn(size(I_G));
I_G(I_G<0) = 0; I_G(I_G>1) = 1;
Ibf_G = bfilter2(I_G,w,sigma);

I_B= I_B+0.03*randn(size(I_B));
I_B(I_B<0) = 0; I_B(I_B>1) = 1;
Ibf_B = bfilter2(I_B,w,sigma);

%两两相减，求得三个抽象差分图
I_R_G  = abs(Ibf_R - Ibf_G);
I_G_B  = abs(Ibf_G -Ibf_B);
I_B_R  = abs(Ibf_B- Ibf_R);

%求均值，得到Imean
Imean = (I_R_G + I_G_B + I_B_R)./3;

%求If

If = zeros(size(I_R));
If = max(Itemp,[],3) - min(Itemp,[],3);

%求最终导向图像Iguidance，通过洗漱啊，集合Imean与If
a = 0.2;
Iguidance = a.*Imean +(1-a).*If;

%导向滤波参数调整
r = 4;
eps = 0.01^2;
q = zeros(size(I));
%进行导向滤波

q(:, :, 1) = guidedfilter(Iguidance, Itemp(:, :, 1), r, eps);
q(:, :, 2) = guidedfilter(Iguidance, Itemp(:, :, 2), r, eps);
q(:, :, 3) = guidedfilter(Iguidance, Itemp(:, :, 3), r, eps);
figure();
imshow(Itemp);
title('原图');
figure;
imshow(Imean);
figure;
imshow(If);
figure;
imshow(Iguidance);
%结果打印输出
figure;
imshow(q);
title('结果图');