function [ img_lab ] = pre_rgb2lab( img_in,max_side )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%img_rgb:���ź��rgbͼ
%img_lab��rgb2lab���ͼ
%h��w��ͼ��߿�



img=double(img_in);


[h_t,w_t]=size(img(:,:,1));%�õ�ԭͼ�ĳ���

if w_t>h_t              %�����Ϳ��ĸ��Ƚϴ�,��ԭͼ��������ֵΪ250
  
    img_rgb(:,:,1)=imresize(img(:,:,1),[(max_side/w_t)*h_t,max_side],'bilinear');%��˫���Բ�ֵ��
    img_rgb(:,:,2)=imresize(img(:,:,2),[(max_side/w_t)*h_t,max_side],'bilinear');
    img_rgb(:,:,3)=imresize(img(:,:,3),[(max_side/w_t)*h_t,max_side],'bilinear');
else
     img_rgb(:,:,1)=imresize(img(:,:,1),[max_side,(max_side/h_t)*w_t],'bilinear');
     img_rgb(:,:,2)=imresize(img(:,:,2),[max_side,(max_side/h_t)*w_t],'bilinear');
     img_rgb(:,:,3)=imresize(img(:,:,3),[max_side,(max_side/h_t)*w_t],'bilinear');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
   
%========================================================   

%=====================RGB2Lab=============================
img_lab_o=RGB2Lab(img_rgb);

img_lab(:,:,1)=img_lab_o(:,:,1)*255/100;      %Lͨ����0��100ת��0��255
img_lab(:,:,2)=(img_lab_o(:,:,2)+120)*255/200;%a��ͨ����-120��120ת��0��255
img_lab(:,:,3)=(img_lab_o(:,:,3)+120)*255/200;

%========================================================
%��һ������0��255�鵽0��1
img_lab=img_lab/255;



end

