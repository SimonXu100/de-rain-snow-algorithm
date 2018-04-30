function [ DistanceValue_t1,DistanceValue_exp,DistanceValue_t1_rang,DistanceValue_exp_rang] = distanceValueMap_search_onescale_2( img_in,max_side )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here


[h,w]=size(img_in(:,:,1));
img_scale1=img_in;

%================================================================================

%�ڼ����֮ǰ�����Ǳ�Ե�㣬�����ȶԾ���������
q=3;%����Ե���ظ���
img_scale1_pad(:,:,1)=padarray(img_scale1(:,:,1),[q,q],'replicate','both');
img_scale1_pad(:,:,2)=padarray(img_scale1(:,:,2),[q,q],'replicate','both');
img_scale1_pad(:,:,3)=padarray(img_scale1(:,:,3),[q,q],'replicate','both');

%----------------------------------------------------------------------------------
%����ó߶��´洢patch�����ľ��󣬸���������ȡһ��patch
for i=1:h
    for j=1:w
        img_scale1_patchVstore_all(i,j,:)=reshape(img_scale1_pad(i:i+6,j:j+6,:),1,147);
    end
end
img_scale1_patchVstore_x=img_scale1_patchVstore_all(1:q:h,1:q:w,:);%������patch��������Ϊһ������

ones_Vstore_temp=ones(size(img_scale1_patchVstore_x));%����ȫһ����



disp('img_scale1_patchVstore_x complete');

%-----------------------------------------------------------------------------------------
%-----------------------------------------------------------------------------------------
%��ʼ����ÿһ��ľ����ֵ
[h_x,w_x]=size(img_scale1_patchVstore_x(:,:,1));%��¼�洢����Ĵ�С

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
             
             %store_o_disvalue_scale1_x(i,j)=norm( store_tp1_all-store_tp1_x); %����mex_store_tp1���غ�����ȫͼ���ؼ����ŷʽ����洢
             
             store_position_dis_x(i,j)=norm([i0,j0]-[(1+(i-1)*3),(1+(j-1)*3)]);
             
          end
     end
     %--------------������ֵ���й�һ��---------�˴������⣬�ǵ�һ���ؽ��еĹ�һ�������Ӧ�øĳ�ȫͼ���رȽϺã���û�и�-----------------------
     max_1=max(max(store_o_disvalue_scale1_x));
       storemap_scale1_x_rang=mat2gray(store_o_disvalue_scale1_x,[0,max_1]); 
       store_position_dis_x_rang=mat2gray( store_position_dis_x,[0,max_side]);%��0�����߳�֮���һ��
       
       %---------�������ֵŷʽ�����λ��ŷʽ���룬����ÿ��patch����i0��j0�����վ���ֵ--------
   
       c=3;%�趨�ù�ʽ�е�һ������
       
       %----------����Ϊ�ó߶��еľ���ֵ
             Dis_scale1=storemap_scale1_x_rang./(1+3* store_position_dis_x_rang);
             
    %------------------�������ó߶ȵľ���ֵ���Ҿ�Ŀ��patchŷʽ������С��65��patch������¼λ�ú�ŷʽ����ֵ--------------- 
    
     Dis_all=Dis_scale1;
     num_K=65;%�ھ�������65����Сֵ
        
        cout=0;%����������ʼֵ
        min_v=[];  %���������ʼֵ
        
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
   %������ֵ���й�һ��
   DistanceValue_t1_rang=mat2gray(DistanceValue_t1);
   DistanceValue_exp_rang=mat2gray(DistanceValue_exp);



end

