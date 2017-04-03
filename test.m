close all;
clear all;
clc;


rng('default')  % For reproducibility
mu1 = [1 2];
sigma1 = [3 .2; .2 2];
mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
X = [mvnrnd(mu1,sigma1,200);mvnrnd(mu2,sigma2,100)];

scatter(X(:,1),X(:,2),10,'ko')


% plot3(X(:,1), X(:,2), X(:,3),'r.');

options = statset('Display', 'final');
gm = fitgmdist(X, 2, 'Options', options);

% hold on;
% ezcontour(@(x, y)pdf(gm, [x y]), [-8 6], [-8 6]);
% hold off;

idx = cluster(gm, X);
cluster1 = (idx == 1);
cluster2 = (idx == 2);

scatter(X(cluster1,1),X(cluster1,2),10,'r+');
hold on
scatter(X(cluster2,1),X(cluster2,2),10,'bo');
hold off
legend('Cluster 1','Cluster 2','Location','NW')

P = posterior(gm,X);

scatter(X(cluster1,1),X(cluster1,2),10,P(cluster1,1),'+')
hold on
scatter(X(cluster2,1),X(cluster2,2),10,P(cluster2,1),'o')
hold off
legend('Cluster 1','Cluster 2','Location','NW')
clrmap = jet(80); colormap(clrmap(9:72,:))
ylabel(colorbar,'Component 1 Posterior Probability')

Y = [10, 10];
pdf = mvnpdf(Y, gm.mu(1, :), gm.Sigma(:, :, 1));
pp = posterior(gm, Y);

[~,order] = sort(P(:,1));
plot(1:size(X,1),P(order,1),'r-',1:size(X,1),P(order,2),'b-');
legend({'Cluster 1 Score' 'Cluster 2 Score'},'location','NW');
ylabel('Cluster Membership Score');
xlabel('Point Ranking');

gm2 = fitgmdist(X,2,'CovType','Diagonal',...
  'SharedCov',true);
P2 = posterior(gm2,X); % equivalently [idx,~,P2] = cluster(gm2,X)
[~,order] = sort(P2(:,1));
plot(1:size(X,1),P2(order,1),'r-',1:size(X,1),P2(order,2),'b-');
legend({'Cluster 1 Score' 'Cluster 2 Score'},'location','NW');
ylabel('Cluster Membership Score');
xlabel('Point Ranking');



load('tmp.mat');

xx = repmat(x, 126, 1);
yy = repmat(y, 126, 1)';
zz = 100 * val;

DT = delaunayTriangulation(xx(:), yy(:), zz(:));
figure
tetramesh(DT, 'FaceAlpha', 0.3);


tic;




% addpath(genpath('../'));
% 
% img_path = '../EM_P.Fua_Mitochondria/';
% training_img_name = 'training.tif';
% training_gd_img_name = 'training_groundtruth.tif';
% testing_img_name = 'testing.tif';
% testing_gd_img_name = 'testing_groundtruth.tif';
% 
% num_training_img = 165;
% num_testing_img = 0;
% imfinfo([img_path testing_img_name]);
% 
% for i = 1 : 1 : 1
%     img1 = imread([img_path training_img_name], i);
%     figure, imshow(img1);
%     img2 = im2double(img1) * 100;
%     sigma = 5;
%     gausFilter = fspecial('gaussian', [10 10], sigma);
%     img_blur=imfilter(img1, gausFilter, 'replicate');
%     img_blur=imfilter(img_blur, gausFilter, 'replicate');
%     img_blur=imfilter(img_blur, gausFilter, 'replicate');
%     img_blur=imfilter(img_blur, gausFilter, 'replicate');
% %     pts = [1 : ];
%     figure, imshow(img_blur);
%     h = figure('position', [50 50 1500 750]);
%     mesh(im2double(img_blur) * 100);
%     
%     colormap(bone);
% end
% 
% tic;