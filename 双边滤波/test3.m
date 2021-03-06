clear
clc

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
[x,y,v] = find(0.1<saliencyMap);
C=zeros(size(saliencyMap));
C(sub2ind(size(C), x, y))=1;
C1 = C.*saliencyMap;
[x,y,v] = find(C1<0.18);
C=zeros(size(C1));
C(sub2ind(size(C), x, y))=1;
C1 = C.*C1;
figure;
imhist(C1);
figure;
imshow(C1);
I= imread('snowybox_small.jpg');
I = rgb2gray(I);
%Thirdlevel = C1.*I;
%figure;
%imshow(uint8(Thirdlevel));

