clear
clc
close all;
%%
%���ߣ� ����ɭ  
%���ʱ�䣺2016��7��21��
%�������������������Լ���ͼ��ʵ�ֻ���˫���˲�����������Ӧ��ȥ��ѩ�㷨
%����������ȥ��Ϊ��
%%

%�������ͼ���������õ�����ͼ
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

% �ҵ�������Ϊ�Ҷȼ���ΧΪ0-0.3�����ص�λ�ã�����ԭͼ�з�����ò���
[x,y,v] = find(saliencyMap< 0.3);
A=zeros(size(saliencyMap));
A(sub2ind(size(A), x, y))=1;
A1 = A.*saliencyMap;
figure;
imhist(A1);
%�ҵ�������Ϊ�Ҷȼ���ΧΪ0.5-1�����ص�λ�ã�����ԭͼ�з�����ò���
[x,y,v] = find(saliencyMap>0.5);
B=zeros(size(saliencyMap));
B(sub2ind(size(B), x, y))=1;
B1= B.*saliencyMap;
figure;
imhist(B1);

%�ҵ�������Ϊ�Ҷȼ���ΧΪ0.3-0.5�����ص�λ�ã�����ԭͼ�з�����ò���
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

%��ԭͼ����з�ΪR,G��B����ͨ��
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
title('ԭͼ');

%�Բ�ͬ��β���ͼ�����в�ͬ������˫���˲�
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

%����������ͼ�ϳ�һ��ͼ������ӡ�����
Result = FirstlevelTemp + SecondlevelTemp + ThirdlevelTemp ;
figure, imshow(Result,[]);