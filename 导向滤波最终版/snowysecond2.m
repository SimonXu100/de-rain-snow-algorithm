clc
clear
close all;
%%
%���ߣ� ����ɭ  
%���ʱ�䣺2016��7��21��
%����������������ѩ��ѧͳ����������ʵ�ֻ��ڵ����˲�����������Ӧ��ȥ��ѩ�㷨
%����������ȥѩΪ��
%%
%����Ҫ�����ͼ�������з�ͨ�����ֱ�õ�����ͨ����ͼ��
I= imread('snowybox_small.jpg');
Itemp = double(I)/255;
I_R = Itemp(:,:,1);
I_G = Itemp(:,:,2);
I_B = Itemp(:,:,3);



w     = 5;       % �˲������
sigma = [3 0.1]; % �����˲�������������ǰ��Ϊsigma��d, sigma-r

%������ͨ���ֱ����˫���˲�
I_R = I_R+0.03*randn(size(I_R));
I_R(I_R<0) = 0; I_R(I_R>1) = 1;
Ibf_R = bfilter2(I_R,w,sigma);


I_G = I_G+0.03*randn(size(I_G));
I_G(I_G<0) = 0; I_G(I_G>1) = 1;
Ibf_G = bfilter2(I_G,w,sigma);

I_B= I_B+0.03*randn(size(I_B));
I_B(I_B<0) = 0; I_B(I_B>1) = 1;
Ibf_B = bfilter2(I_B,w,sigma);

%����������������������ͼ
I_R_G  = abs(Ibf_R - Ibf_G);
I_G_B  = abs(Ibf_G -Ibf_B);
I_B_R  = abs(Ibf_B- Ibf_R);

%���ֵ���õ�Imean
Imean = (I_R_G + I_G_B + I_B_R)./3;

%��If

If = zeros(size(I_R));
If = max(Itemp,[],3) - min(Itemp,[],3);

%�����յ���ͼ��Iguidance��ͨ��ϴ����������Imean��If
a = 0.2;
Iguidance = a.*Imean +(1-a).*If;

%�����˲���������
r = 4;
eps = 0.01^2;
q = zeros(size(I));
%���е����˲�

q(:, :, 1) = guidedfilter(Iguidance, Itemp(:, :, 1), r, eps);
q(:, :, 2) = guidedfilter(Iguidance, Itemp(:, :, 2), r, eps);
q(:, :, 3) = guidedfilter(Iguidance, Itemp(:, :, 3), r, eps);
figure();
imshow(Itemp);
title('ԭͼ');
figure;
imshow(Imean);
figure;
imshow(If);
figure;
imshow(Iguidance);
%�����ӡ���
figure;
imshow(q);
title('���ͼ');