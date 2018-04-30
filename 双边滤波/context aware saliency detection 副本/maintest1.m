%====test1
clear
close all
clc

%img_in,����Ĵ������ͼ��
%n��ͼ��ֿ��patch�Ĵ�С


%img_rgb:���ź��rgbͼ
%img_lab��rgb2lab���ͼ
%h��w��ͼ��߿�
%mex_store,�洢����


img=imread('bird.jpg');
%img=imread('E:\����\matlab\�����Էָ�\������֪��ֵ\test picture\0010.tif');
max_side=120;
yu_value=0.8;

img_lab = pre_rgb2lab(img,max_side);

[ img_scale_1,img_scale_2,img_scale_3,img_scale_4 ] = get4Scale( img_lab );

[ DistanceValue_scale_1_t1,DistanceValue_scale_1_exp,DistanceValue_scale_1_t1_rang,DistanceValue_scale_1_exp_rang] = distanceValueMap_search_onescale_2( img_scale_1,max_side );
[ DistanceValue_scale_2_t1,DistanceValue_scale_2_exp,DistanceValue_scale_2_t1_rang,DistanceValue_scale_2_exp_rang] = distanceValueMap_search_onescale_2( img_scale_2,max_side );
[ DistanceValue_scale_3_t1,DistanceValue_scale_3_exp,DistanceValue_scale_3_t1_rang,DistanceValue_scale_3_exp_rang] = distanceValueMap_search_onescale_2( img_scale_3,max_side );
[ DistanceValue_scale_4_t1,DistanceValue_scale_4_exp,DistanceValue_scale_4_t1_rang,DistanceValue_scale_4_exp_rang] = distanceValueMap_search_onescale_2( img_scale_4,max_side );

value_C_1=Center_weight( DistanceValue_scale_1_exp_rang,yu_value );
value_C_2=Center_weight( DistanceValue_scale_2_exp_rang,yu_value );
value_C_3=Center_weight( DistanceValue_scale_3_exp_rang,yu_value );
value_C_4=Center_weight( DistanceValue_scale_4_exp_rang,yu_value );

[h,w]=size(value_C_1);
value_C_1_resize=value_C_1;
value_C_2_resize=imresize(value_C_2,[h,w]);
value_C_3_resize=imresize(value_C_3,[h,w]);
value_C_4_resize=imresize(value_C_4,[h,w]);

value_C_sum=(value_C_1_resize+value_C_2_resize+value_C_3_resize+value_C_4_resize)/4;
figure,
imshow(value_C_sum);
figure,
imshow(value_C_1_resize);
figure,
imshow(value_C_2_resize);   
figure,
imshow(value_C_3_resize);
figure,
imshow(value_C_4_resize);

