clc
clear
close all;

I= imread('snowybox_small.jpg');
Itemp = double(I)/255;
I_R = Itemp(:,:,1);
I_G = Itemp(:,:,2);
I_B = Itemp(:,:,3);
%figure;
%imshow(rgb2gray(I));


w     = 1;       % bilateral filter half-width
sigma = [1 1]; % bilateral filter standard deviations

I_R = I_R+0.03*randn(size(I_R));
I_R(I_R<0) = 0; I_R(I_R>1) = 1;
Ibf_R = bfilter2(I_R,w,sigma);


I_G = I_G+0.03*randn(size(I_G));
I_G(I_G<0) = 0; I_G(I_G>1) = 1;
Ibf_G = bfilter2(I_G,w,sigma);

I_B= I_B+0.03*randn(size(I_B));
I_B(I_B<0) = 0; I_B(I_B>1) = 1;
Ibf_B = bfilter2(I_B,w,sigma);

I_R_G  = abs(Ibf_R - Ibf_G);
I_G_B  = abs(Ibf_G -Ibf_B);
I_B_R  = abs(Ibf_B- Ibf_R);
Imean = (I_R_G + I_G_B + I_B_R)./3;


If = zeros(size(I_R));
[x,y] = size(If);
for i = 1:x
   for j = 1:y
       If(i,j) =max([I_R(i,j),I_G(i,j),I_B(i,j)]) - min([I_R(i,j),I_G(i,j),I_B(i,j)]);
   end
end

a = 0.2;
Iguidance = a.*Imean +(1-a).*If;
%Iguidance = Iguidance+0.03*randn(size(Iguidance));
%Iguidance(Iguidance<0) = 0; Iguidance(Iguidance>1) = 1;
%Iguidance = bfilter2(Iguidance,w,sigma);


%Iguidance = a.*Imean +(1-a).*If;
r =8;
eps = 0.05^2;
q = zeros(size(I));


q(:, :, 1) = guidedfilter(Iguidance, Itemp(:, :, 1), r, eps);
q(:, :, 2) = guidedfilter(Iguidance, Itemp(:, :, 2), r, eps);
q(:, :, 3) = guidedfilter(Iguidance, Itemp(:, :, 3), r, eps);
figure();
imshow(Itemp);
figure;
imshow(Imean);
figure;
imshow(If);
figure;
imshow(Iguidance);
figure;
imshow(q);