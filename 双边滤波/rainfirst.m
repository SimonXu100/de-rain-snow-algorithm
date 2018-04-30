clear
clc
close all;
%%
%作者： 徐书森  
%完成时间：2016年7月21日
%功能描述：利用显著性检验图，实现基于双边滤波的内容自适应的去雨雪算法
%本程序中以去雨为例
%%

%读入对雨图像，事先做好的显著图
saliencyMap = double(imread('SaliencyMap2.jpg'))/255;
%%inImg = imresize(inImg, 64/size(inImg, 2));
figure;
imhist(saliencyMap);
figure;
imshow(saliencyMap,[]);
saliencyMap=imresize(saliencyMap,[240,360]);
figure;
imhist(saliencyMap);
figure;
imshow(saliencyMap,[]);

% 找到显著性为灰度级范围为0-0.3的像素点位置，并从原图中分离出该补丁
[x,y,v] = find(saliencyMap< 0.3);
A=zeros(size(saliencyMap));
A(sub2ind(size(A), x, y))=1;
A1 = A.*saliencyMap;
figure;
imhist(A1);
%找到显著性为灰度级范围为0.5-1的像素点位置，并从原图中分离出该补丁
[x,y,v] = find(saliencyMap>0.5);
B=zeros(size(saliencyMap));
B(sub2ind(size(B), x, y))=1;
B1= B.*saliencyMap;
figure;
imhist(B1);

%找到显著性为灰度级范围为0.3-0.5的像素点位置，并从原图中分离出该补丁
[x,y,v] = find(0.3<saliencyMap);
C1=zeros(size(saliencyMap));
C1(sub2ind(size(C1), x, y))=1;
Cimage1 = C1.*saliencyMap;
figure;
imhist(Cimage1);
[x,y,v] = find(Cimage1<0.5);
C2=zeros(size(Cimage1));
C2(sub2ind(size(C2), x, y))=1;
Cimage = C2.*Cimage1;
C = C1 .* C2;
figure;
imhist(Cimage);
figure;
I= imread('forrest_small.jpg');
Itemp = double(I);

%对原图像进行分为R,G，B三个通道
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
title('原图');

%对不同层次补丁图，进行不同参数的双边滤波
Thirdlevel = double(Thirdlevel)/255;  
Thirdlevel = Thirdlevel +0.03*randn(size(Thirdlevel ));
Thirdlevel (Thirdlevel <0) = 0; Thirdlevel (Thirdlevel>1) = 1;
ThirdlevelTemp = bfilter2(Thirdlevel,2.5,[0.6,0.02]);
figure, imshow(ThirdlevelTemp,[]); 

Secondlevel = double(Secondlevel)/255;  
Secondlevel =Secondlevel +0.03*randn(size(Secondlevel ));
Secondlevel(Secondlevel<0) = 0; Secondlevel(Secondlevel>1) = 1;
SecondlevelTemp  = bfilter2(Secondlevel,5,[ 3, 0.1]);
figure, imshow(SecondlevelTemp,[]); 

Firstlevel = double(Firstlevel)/255;
Firstlevel = Firstlevel +0.03*randn(size(Firstlevel));
Firstlevel(Firstlevel<0) = 0;Firstlevel(Firstlevel>1) = 1;
FirstlevelTemp= bfilter2(Firstlevel,5,[3 0.1]);
figure, imshow(FirstlevelTemp,[]); 

%把三个补丁图合成一个图，并打印输出。
Result = FirstlevelTemp + SecondlevelTemp + ThirdlevelTemp ;
figure, imshow(Result,[]);