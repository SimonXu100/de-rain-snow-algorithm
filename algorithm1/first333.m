clear
clc
close all;
%% Read image from file
saliencyMap = double(imread('SaliencyMap.jpg'))/255;
%%inImg = imresize(inImg, 64/size(inImg, 2));
figure;
imhist(saliencyMap);
figure;

imshow(saliencyMap,[]);
saliencyMap=imresize(saliencyMap,[240,359]);
figure;
imhist(saliencyMap);
figure;
imshow(saliencyMap,[]);
[x,y,v] = find(saliencyMap< 0.35);
A=zeros(size(saliencyMap));
A(sub2ind(size(A), x, y))=1;
A1 = A.*saliencyMap;
figure;
imhist(A1);
%figure;
%imshow(A1);
[x,y,v] = find(saliencyMap>0.7);
B=zeros(size(saliencyMap));
B(sub2ind(size(B), x, y))=1;
B1= B.*saliencyMap;
figure;
imhist(B1);
[x,y,v] = find(0.35<saliencyMap);
C1=zeros(size(saliencyMap));
C1(sub2ind(size(C1), x, y))=1;
Cimage1 = C1.*saliencyMap;
figure;
imhist(Cimage1);
[x,y,v] = find(Cimage1<0.7);
C2=zeros(size(Cimage1));
C2(sub2ind(size(C2), x, y))=1;
Cimage = C2.*Cimage1;
C = C1 .* C2;
figure;
imhist(Cimage);
figure;
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
title('hagkjhkjghjd');
Thirdlevel = double(Thirdlevel)/255;  
Thirdlevel  = Thirdlevel+0.05*randn(size(Thirdlevel ));  
Thirdlevel (Thirdlevel<0) = 0;  Thirdlevel(Thirdlevel >1) = 1;  
ThirdlevelTemp = BilateralFilt2(Thirdlevel,5,[6 0.2]);
figure, imshow(ThirdlevelTemp,[]); 

Secondlevel= double(Secondlevel)/255;  
Secondlevel = Secondlevel+0.05*randn(size(Secondlevel ));  
Secondlevel (Secondlevel<0) = 0;  Secondlevel(Secondlevel >1) = 1;  
SecondlevelTemp = BilateralFilt2(Secondlevel,25,[6 0.2]);
figure, imshow(SecondlevelTemp,[]); 

Firstlevel= double(Firstlevel)/255;  
Firstlevel = Firstlevel+0.05*randn(size(Firstlevel ));  
Firstlevel (Firstlevel<0) = 0; Firstlevel(Firstlevel >1) = 1;  
FirstlevelTemp = BilateralFilt2(Firstlevel,25,[6 0.2]);
figure, imshow(FirstlevelTemp,[]); 
Result = FirstlevelTemp + SecondlevelTemp + ThirdlevelTemp ;
%Result = BilateralFilt2(Result,5,[3 0.1]);
figure, imshow(Result,[]);


