function [ DistanceValue_t1,DistanceValue_exp,DistanceValue_t1_rang,DistanceValue_exp_rang] = distanceValueMap_search_onescale_2( img_in,max_side )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here


[h,w]=size(img_in(:,:,1));
img_scale1=img_in;

%================================================================================

%在计算块之前，考虑边缘点，所以先对矩阵进行填充
q=3;%填充边缘像素个数
img_scale1_pad(:,:,1)=padarray(img_scale1(:,:,1),[q,q],'replicate','both');
img_scale1_pad(:,:,2)=padarray(img_scale1(:,:,2),[q,q],'replicate','both');
img_scale1_pad(:,:,3)=padarray(img_scale1(:,:,3),[q,q],'replicate','both');

%----------------------------------------------------------------------------------
%计算该尺度下存储patch相量的矩阵，各三个像素取一个patch
for i=1:h
    for j=1:w
        img_scale1_patchVstore_all(i,j,:)=reshape(img_scale1_pad(i:i+6,j:j+6,:),1,147);
    end
end
img_scale1_patchVstore_x=img_scale1_patchVstore_all(1:q:h,1:q:w,:);%将代表patch的相量存为一个矩阵

ones_Vstore_temp=ones(size(img_scale1_patchVstore_x));%生成全一矩阵



disp('img_scale1_patchVstore_x complete');

%-----------------------------------------------------------------------------------------
%-----------------------------------------------------------------------------------------
%开始计算每一点的距离的值
[h_x,w_x]=size(img_scale1_patchVstore_x(:,:,1));%记录存储矩阵的大小

for i0=1:h
    
    disp(num2str(i0));
    
    for j0=1:w
     
    % i0=1;
    % j0=1;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     temp1=img_scale1_patchVstore_all(i0,j0,:);
     
     for i2=1:length(temp1)
         temp_x_1(:,:,i2)=temp1(:,:,i2)*ones_Vstore_temp(:,:,i2);
     end
    
     temp_x_2=temp_x_1-img_scale1_patchVstore_x;
     temp_x_2=temp_x_2.*temp_x_2;
     sum_temp_x=sum(temp_x_2,3);
     
    store_o_disvalue_scale1_x= sum_temp_x.^0.5;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
    
      %temp1=img_scale1_patchVstore_all(i0,j0,:);
     % store_tp1_all=reshape(img_scale1_patchVstore_all(i0,j0,:),1,147);
      
     for i=1:h_x
         for j=1:w_x
             
             %store_tp1_x=reshape(img_scale1_patchVstore_x(i,j,:),1,147);
             
             %store_o_disvalue_scale1_x(i,j)=norm( store_tp1_all-store_tp1_x); %将与mex_store_tp1像素和其他全图像素计算的欧式距离存储
             
             store_position_dis_x(i,j)=norm([i0,j0]-[(1+(i-1)*3),(1+(j-1)*3)]);
             
          end
     end
     %--------------将距离值进行归一化---------此处有问题，是单一像素进行的归一化，这个应该改成全图像素比较好，先没有改-----------------------
     max_1=max(max(store_o_disvalue_scale1_x));
       storemap_scale1_x_rang=mat2gray(store_o_disvalue_scale1_x,[0,max_1]); 
       store_position_dis_x_rang=mat2gray( store_position_dis_x,[0,max_side]);%在0到最大边长之间归一化
       
       %---------结合相量值欧式距离和位置欧式距离，计算每个patch对于i0，j0的最终距离值--------
   
       c=3;%设定该公式中的一个参数
       
       %----------以下为该尺度中的距离值
             Dis_scale1=storemap_scale1_x_rang./(1+3* store_position_dis_x_rang);
             
    %------------------在上述该尺度的距离值中找距目标patch欧式距离最小的65个patch，并记录位置和欧式距离值--------------- 
    
     Dis_all=Dis_scale1;
     num_K=65;%在矩阵中找65个最小值
        
        cout=0;%计数变量初始值
        min_v=[];  %计数矩阵初始值
        
while cout<num_K
    mm=min(min(Dis_all));
    f=find(Dis_all==mm);
    min_v=[min_v,Dis_all(f)'];
    cout=length(min_v);
    Dis_all(f)=10;
end
    DistanceValue_t1(i0,j0)=sum(min_v)/num_K;
    DistanceValue_exp(i0,j0)=1-exp(- DistanceValue_t1(i0,j0));
    
        
    end
end
   %对两个值进行归一化
   DistanceValue_t1_rang=mat2gray(DistanceValue_t1);
   DistanceValue_exp_rang=mat2gray(DistanceValue_exp);



end

