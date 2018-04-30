function [ img_out ] = Center_weight( img_in,yu_value )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
img_in=mat2gray(img_in);
[h,w]=size(img_in);
[row,col]=find(img_in>yu_value);%找到所有大于阈值的点，记录坐标
l=length(row);
a(1,:)=row;
a(2,:)=col;

for i0=1:h
    for j0=1:w
        
      for i=1:l
        c_store(i)=norm(a(:,i)-[i0;j0]);
      end

      min_c(i0,j0)=min(c_store);
      
    end
end
   min_c_rang=mat2gray(min_c);
   img_out=img_in.*(1-min_c_rang);



end

