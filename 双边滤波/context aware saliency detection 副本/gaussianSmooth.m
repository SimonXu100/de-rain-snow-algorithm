function [result ] = gaussianSmooth( img )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%��ͼ����и�˹ƽ�����������ɸ�˹�������Ȼ���ȥ��
%���ú���������������ƽ��
%-------------------------------------------------------

filter = [1,5,10,10,5,1,1]; 
filter = filter/sum(filter);
[h,w] = size(img);

if min(h,w) > 3
  % big image - use straight-forward method
  result = conv2(filter,filter,img,'same');
  
 % �����н�����
  
  %horResult = convResult(:,[2:2:end]-1);
 % result = horResult([2:2:end]-1,:);
else
  % Image is small along at least one dimension.
  error('������и�˹ƽ����ͼ̫С');
end

end

