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
A1 = A.*saliencyMap;
figure;
imhist(A1);
figure;
imshow(A1);
I= imread('snowybox_small.jpg');
I = rgb2gray(double(I));
Firstlevel = A.*I;
%figure;
%imshow(uint8(Firstlevel));


