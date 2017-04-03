close all;
clear all;
clc;

img_dir = './';
n_gm = 12;

rng('default');  % For reproducibility

for i = 1 : 400
    if exist([img_dir, 'sample_10/pos/sample_10_normal_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_10/pos/sample_10_normal_', num2str(i), '.jpg']));
        scale(1).sample(2 * i - 1, :) = img(:);
        img = fliplr(img);
        scale(1).sample(end + 1, :) = img(:);
    end
end

for i = 1 : 400
    if exist([img_dir, 'sample_10/pos/sample_10_mild_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_10/pos/sample_10_mild_', num2str(i), '.jpg']));
        scale(1).sample(end + 1, :) = img(:);
        img = fliplr(img);
        scale(1).sample(end + 1, :) = img(:);
    end
end

for i = 1 : 400
    if exist([img_dir, 'sample_10/pos/sample_10_moderate_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_10/pos/sample_10_moderate_', num2str(i), '.jpg']));
        scale(1).sample(end + 1, :) = img(:);
        img = fliplr(img);
        scale(1).sample(end + 1, :) = img(:);
    end
end

for i = 1 : 400
    if exist([img_dir, 'sample_10/pos/sample_10_severe_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_10/pos/sample_10_severe_', num2str(i), '.jpg']));
        scale(1).sample(end + 1, :) = img(:);
        img = fliplr(img);
        scale(1).sample(end + 1, :) = img(:);
    end
end

for i = 1 : 400
    if exist([img_dir, 'sample_20/pos/sample_20_normal_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_20/pos/sample_20_normal_', num2str(i), '.jpg']));
        scale(2).sample(2 * i - 1, :) = img(:);
        img = fliplr(img);
        scale(2).sample(end + 1, :) = img(:);
    end
end

for i = 1 : 400
    if exist([img_dir, 'sample_20/pos/sample_20_mild_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_20/pos/sample_20_mild_', num2str(i), '.jpg']));
        scale(2).sample(end + 1, :) = img(:);
        img = fliplr(img);
        scale(2).sample(end + 1, :) = img(:);
    end
end

for i = 1 : 400
    if exist([img_dir, 'sample_20/pos/sample_20_moderate_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_20/pos/sample_20_moderate_', num2str(i), '.jpg']));
        scale(2).sample(end + 1, :) = img(:);
        img = fliplr(img);
        scale(2).sample(end + 1, :) = img(:);
    end
end

for i = 1 : 400
    if exist([img_dir, 'sample_20/pos/sample_20_severe_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_20/pos/sample_20_severe_', num2str(i), '.jpg']));
        scale(2).sample(end + 1, :) = img(:);
        img = fliplr(img);
        scale(2).sample(end + 1, :) = img(:);
    end
end

[~, scale(2).sample_pca] = pca(scale(2).sample, 'NumComponents', 16);
[~, scale(1).sample_pca] = pca(scale(1).sample, 'NumComponents', 8);

n_scale = 2;
n_gmm = 12;

for i = 1 : n_scale
%     scale(i).sample = scale(i).sample_pca * 256;
    scale(i).gmm = fitgmdist(scale(i).sample_pca, 12, 'Regularize', 0.01);
    scale(i).sample_post = posterior(scale(i).gmm, scale(i).sample_pca);
    for j = 1 : n_gmm
        scale(i).tp_prob(j, :) = scale(i).sample_post(:, j) .* mvnpdf(scale(i).sample_pca(:, :), scale(i).gmm.mu(j, :), scale(i).gmm.Sigma(:, :, j));
    end
    scale(i).prob = -log(sum(scale(i).tp_prob, 1));
    scale(i).v_gm = fitgmdist(scale(i).prob', 1);
    scale(i).maxv = scale(i).v_gm.mu + 2 * scale(i).v_gm.Sigma;
end

% [~, score] = pca(sample, 'NumComponents', 2);
% 
% species = zeros(size(score, 1), 1);
% 
% gscatter(score(:,1),score(:,2),species);
% hold on;
% 
% options = statset('Display', 'final');
% gm = fitgmdist(score, 256, 'Regularize', 0.01);
% % gm = fitgmdist(sample, 4, 'Options', options);

% idx = cluster(gm, sample);
% for i = 1 : 12
%     cluster(i) = length(find(idx == i));
% end
% post = posterior(gm, sample(:, :));
% pst = max(post, [], 2);
% tic;

test_img_dir = './sample_10/neg/';
% for i = 1 : 400
%     if exist([img_dir, 'sample_10/neg/sample_10_normal_', num2str(i), '.jpg'])
%         img = im2double(imread([img_dir, 'sample_10/neg/sample_10_normal_', num2str(i), '.jpg']));
%         test(2 * i - 1, :) = img(:);
%         img = fliplr(img);
%         test(end + 1, :) = img(:);
%     end
% end
% for i = 1 : 400
%     if exist([img_dir, 'sample_10/neg/sample_10_mild_', num2str(i), '.jpg'])
%         img = im2double(imread([img_dir, 'sample_10/neg/sample_10_mild_', num2str(i), '.jpg']));
%         test(end + 1, :) = img(:);
%         img = fliplr(img);
%         test(end + 1, :) = img(:);
%     end
% end
% for i = 1 : 400
%     if exist([img_dir, 'sample_10/neg/sample_10_moderate_', num2str(i), '.jpg'])
%         img = im2double(imread([img_dir, 'sample_10/neg/sample_10_moderate_', num2str(i), '.jpg']));
%         test(end + 1, :) = img(:);
%         img = fliplr(img);
%         test(end + 1, :) = img(:);
%     end
% end
cnt = 0;
% for i = 1 : 400
%     if exist([img_dir, 'sample_10/neg/sample_20_severe_', num2str(i), '.jpg'])
%         img = im2double(imread([img_dir, 'sample_10/neg/sample_20_severe_', num2str(i), '.jpg']));
%         cnt = cnt + 1;
%         test(cnt, :) = img(:);
%         img = fliplr(img);
%         cnt = cnt + 1;
%         test(cnt, :) = img(:);
%     end
% end
for i = 1 : 400
    if exist([img_dir, 'sample_10/pos_test/sample_10_normal_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_10/pos_test/sample_10_normal_', num2str(i), '.jpg']));
        cnt = cnt + 1;
        test(cnt, :) = img(:);
        img = fliplr(img);
        cnt = cnt + 1;
        test(cnt, :) = img(:);
    end
end
for i = 1 : 400
    if exist([img_dir, 'sample_10/pos_test/sample_10_mild_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_10/pos_test/sample_10_mild_', num2str(i), '.jpg']));
        cnt = cnt + 1;
        test(cnt, :) = img(:);
        img = fliplr(img);
        cnt = cnt + 1;
        test(cnt, :) = img(:);
    end
end
for i = 1 : 400
    if exist([img_dir, 'sample_10/pos_test/sample_10_moderate_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_10/pos_test/sample_10_moderate_', num2str(i), '.jpg']));
        cnt = cnt + 1;
        test(cnt, :) = img(:);
        img = fliplr(img);
        cnt = cnt + 1;
        test(cnt, :) = img(:);
    end
end
for i = 1 : 400
    if exist([img_dir, 'sample_10/pos_test/sample_10_severe_', num2str(i), '.jpg'])
        img = im2double(imread([img_dir, 'sample_10/pos_test/sample_10_severe_', num2str(i), '.jpg']));
        cnt = cnt + 1;
        test(cnt, :) = img(:);
        img = fliplr(img);
        cnt = cnt + 1;
        test(cnt, :) = img(:);
    end
end

[~, test_pca] = pca(test, 'NumComponents', 8);

% test = test * 256;

% species = zeros(size(sample, 1), 1);
% 
% [~, score] = pca(sample, 'NumComponents', 3);
% 
% plot3(score(:, 1), score(:, 2), score(:, 3), 'r.');
% hold on;

% sample = [sample; test];
% 
% [~, score] = pca(test, 'NumComponents', 3);
% 
% plot3(score(:, 1), score(:, 2), score(:, 3), 'b.');
% 
% species = [species; ones(size(test, 1), 1)];
% 
% gscatter(score(:,1),score(:,2),species);

test_post = posterior(scale(1).gmm, test_pca(:, :));

for j = 1 : n_gmm
    tp_prob(j, :) = test_post(:, j) .* mvnpdf(test_pca(:, :), scale(1).gmm.mu(j, :), scale(1).gmm.Sigma(:, :, j));
end
test_prob = -log(sum(tp_prob, 1));

tic;

% mu1 = [5 5 5];
% sigma1 = [1 0 0; 0 1 0; 0 0 1];
% mu2 = [-5 -5 -5];
% sigma2 = [1 0 0; 0 1 0; 0 0 1];
% X = [mvnrnd(mu1,sigma1,200); mvnrnd(mu2,sigma2,100)];

plot3(X(:,1), X(:,2), X(:,3),'r.');

options = statset('Display', 'final');
gm = fitgmdist(X, 2, 'Options', options);

% hold on;
% ezcontour(@(x, y)pdf(gm, [x y]), [-8 6], [-8 6]);
% hold off;

idx = cluster(gm, X);
cluster1 = (idx == 1);
cluster2 = (idx == 2);

plot3(X(cluster1, 1), X(cluster1,2), X(cluster1,3), 'r+');
hold on
plot3(X(cluster2, 1), X(cluster2,2), X(cluster2,3), 'bo');
hold off
legend('Cluster 1', 'Cluster 2', 'Location', 'NW');

P = posterior(gm,X);

scatter(X(cluster1,1),X(cluster1,2),10,P(cluster1,1),'+')
hold on
scatter(X(cluster2,1),X(cluster2,2),10,P(cluster2,1),'o')
hold off
legend('Cluster 1','Cluster 2','Location','NW')
clrmap = jet(80); colormap(clrmap(9:72,:))
ylabel(colorbar,'Component 1 Posterior Probability')

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