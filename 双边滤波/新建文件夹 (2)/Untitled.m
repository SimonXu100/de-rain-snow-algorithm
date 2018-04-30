clear all;
close all;

%% load parameters and images


file_names{1} = 'snowybox_small.jpg';
MOV = saliency(file_names);
N = length(MOV);
for i=1:N
    figure(i); clf;
    subplot(1,2,1); imshow(MOV{i}.Irgb); title('Input','fontsize',16);
    subplot(1,2,2); imshow(MOV{i}.SaliencyMap); title('Saliency map','fontsize',16);
end   
L = MOV{i}.SaliencyMap;
figure;
imhist(L);
