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
saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [10, 10], 2.5)));figure;
imshow(saliencyMap);
figure;
imhist(saliencyMap);
[x,y,v] = find(saliencyMap>0.18);
B=zeros(size(saliencyMap));
B(sub2ind(size(B), x, y))=1;
B1 = B.*saliencyMap;
figure;
imhist(B1);
figure;
imshow(B1);
I= imread('snowybox_small.jpg');
Secondlevel = double(I);
Secondlevel(:,:,1) = B .* Secondlevel(:,:,1);
Secondlevel(:,:,2) = B .* Secondlevel(:,:,2);
Secondlevel(:,:,3) = B .* Secondlevel(:,:,3);
figure;
imshow(uint8(Secondlevel));
