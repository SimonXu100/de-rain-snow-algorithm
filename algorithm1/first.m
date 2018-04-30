clear
clc
close all;
%% Read image from file
inImg = im2double(rgb2gray(imread('snowybox_small.jpg')));
%%inImg = imresize(inImg, 64/size(inImg, 2));

%% Spectral Residual
myFFT = fft2(inImg);
myLogAmplitude = log(abs(myFFT));
myPhase = angle(myFFT);
mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate');
saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;

%% After Effect
saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [10, 10], 2.5)));
figure;
imshow(saliencyMap);
figure;
imhist(saliencyMap);
[x,y,v] = find(saliencyMap< 0.1);
A=zeros(size(saliencyMap));
A(sub2ind(size(A), x, y))=1;
%figure;
%imshow(A1);
[x,y,v] = find(saliencyMap>0.18);
B=zeros(size(saliencyMap));
B(sub2ind(size(B), x, y))=1;
[x,y,v] = find(0.1<saliencyMap);
C1=zeros(size(saliencyMap));
C1(sub2ind(size(C1), x, y))=1;
Cimage1 = C1.*saliencyMap;
figure;
imhist(Cimage1);
[x,y,v] = find(Cimage1<0.18);
C2=zeros(size(Cimage1));
C2(sub2ind(size(C2), x, y))=1;
%Cimage = C2.*Cimage1;
C = C1 .* C2;
%figure;
%imhist(Cimage);
I= imread('snowybox_small.jpg');
Itemp = double(I);
Firstlevel(:,:,1) = A .* Itemp(:,:,1);
Firstlevel(:,:,2) = A .* Itemp(:,:,2);
Firstlevel(:,:,3) = A .* Itemp(:,:,3);
Secondlevel(:,:,1) = C .* Itemp(:,:,1);
Secondlevel(:,:,2) = C .* Itemp(:,:,2);
Secondlevel(:,:,3) = C .* Itemp(:,:,3);
Thirdlevel(:,:,1) = B .* Itemp(:,:,1);
Thirdlevel(:,:,2) = B .* Itemp(:,:,2);
Thirdlevel(:,:,3) = B .* Itemp(:,:,3);
figure;
imshow(uint8(Firstlevel));
figure;
imshow(uint8(Secondlevel));
figure;
imshow(uint8(Thirdlevel));
H = Firstlevel+ Secondlevel+Thirdlevel;
figure;
imshow(uint8(H));
Thirdlevel = double(Thirdlevel)/255;  
Thirdlevel  = Thirdlevel+0.05*randn(size(Thirdlevel ));  
Thirdlevel (Thirdlevel<0) = 0;  Thirdlevel(Thirdlevel >1) = 1;  
ThirdlevelTemp = BilateralFilt2(Thirdlevel,2.5,[1.5,0.05]);
figure, imshow(ThirdlevelTemp,[]); 

Secondlevel= double(Secondlevel)/255;  
Secondlevel = Secondlevel+0.05*randn(size(Secondlevel ));  
Secondlevel (Secondlevel<0) = 0;  Secondlevel(Secondlevel >1) = 1;  
SecondlevelTemp = BilateralFilt2(Secondlevel,5,[ 3, 0.1]);
figure, imshow(SecondlevelTemp,[]); 

Firstlevel= double(Firstlevel)/255;  
Firstlevel = Firstlevel+0.05*randn(size(Firstlevel ));  
Firstlevel (Firstlevel<0) = 0; Firstlevel(Firstlevel >1) = 1;  
FirstlevelTemp = BilateralFilt2(Firstlevel,10,[6 0.2]);
figure, imshow(FirstlevelTemp,[]); 
Result = FirstlevelTemp + SecondlevelTemp + ThirdlevelTemp ;
figure, imshow(Result,[]);


