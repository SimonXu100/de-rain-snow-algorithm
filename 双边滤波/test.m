img = imread('0004.png');  
%%img = imread('.\images\lena_gray.tif');  
img = double(img)/255;  
img = img+0.05*randn(size(img));  
img(img<0) = 0; img(img>1) = 1;  
%img = imnoise(img,'gaussian');  
figure, imshow(img,[])  
title('Ô­Ê¼Í¼Ïñ')  
d = 20;  
sigma = [3 0.1];  
resultI = BilateralFilt2(double(img), d, sigma);    
figure, imshow(resultI,[]);  
title('Ë«±ßÂË²¨ºóµÄÍ¼Ïñ') 