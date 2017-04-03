% close all;
clear all;
clc;

global class;
global scale_size;
global img_root_dir;
global n_gmm;
global n_scale;
global n_class;
global PCA_dim;
global scale;
global step_size;
global novelty_map;
global test_samples;

step_size = 5;
scale_size = [10, 20];
img_root_dir = './';
n_gmm = 12; % num of GMM
n_scale = 2; % 10*10, 20*20
class(1).name = 'normal';
class(2).name = 'mild';
class(3).name = 'moderate';
class(4).name = 'severe';
PCA_dim = 5;
n_class = 4;

% Initialization();

Train_GMM();

Test_GMM_1();

tic;
