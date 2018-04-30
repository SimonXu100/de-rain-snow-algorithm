function [result ] = gaussianSmooth( img )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%将图像进行高斯平滑，用于生成高斯金字塔等或是去噪
%采用横向，纵向两个向量平滑
%-------------------------------------------------------

filter = [1,5,10,10,5,1,1]; 
filter = filter/sum(filter);
[h,w] = size(img);

if min(h,w) > 3
  % big image - use straight-forward method
  result = conv2(filter,filter,img,'same');
  
 % 不进行降采样
  
  %horResult = convResult(:,[2:2:end]-1);
 % result = horResult([2:2:end]-1,:);
else
  % Image is small along at least one dimension.
  error('输入进行高斯平滑的图太小');
end

end

