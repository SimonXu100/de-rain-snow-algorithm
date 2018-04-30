clc
clear


I= imread('snowybox_small.jpg');
Itemp = double(I);
I_R = Itemp(:,:,1);
I_G = Itemp(:,:,2);
I_B = Itemp(:,:,3);
%figure;
%imshow(rgb2gray(I));
I_R = double(I_R)/255;  
I_R= I_R+0.05*randn(size(I_R));  
I_R (I_R<0) = 0;  I_R(I_R >1) = 1;  
Ibf_R = BilateralFilt2(I_R,5,[3,0.1]);
%figure, imshow(Ibf_R ,[]); 

I_G = double(I_G)/255;  
I_G= I_G+0.05*randn(size(I_G));  
I_G (I_G<0) = 0;  I_G(I_G >1) = 1;  
Ibf_G = BilateralFilt2(I_G,5,[3,0.1]);
%figure, imshow(Ibf_G ,[]); 

I_B = double(I_B)/255;  
I_B= I_B+0.05*randn(size(I_B));  
I_B (I_B<0) = 0;  I_B(I_B >1) = 1;  
Ibf_B = BilateralFilt2(I_B,5,[3,0.1]);
%figure, imshow(Ibf_B ,[]); 

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
a = 0.8;
Iguidance = Imean +If;
%Iguidance = a.*Imean +(1-a).*If;
r =8;
eps = 0.05^2;
Itemp = Itemp/255;
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
