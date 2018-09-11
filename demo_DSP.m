% Example code for ECCV18 paper: 
% Learning Discriminative Video Representations Using Adversarial Perturbations. 
% (https://arxiv.org/pdf/1807.09380.pdf)
% Author: Jue Wang; Anoop Cherian;
addpath(genpath('./manopt/'));
%key=1; randn('state', key); rand('twister', key);
warning 'off';
d=4096; % size of features
n=5; % num seqs.

% generate some data.
tic;
close all;
seqs = cell(n,1);
for t=1:n   
    num_frames = 100; %randi(100);
    X = rand(num_frames, d);  
    X = normr(X); 
    X = cumsum(X,1); 
    X = bsxfun(@rdivide, X, [1:num_frames]');
    seqs{t} = X;
end

% generate some noise vectors
% random noise used here, for UAP please refer to: https://github.com/LTS4/universal
for t=1:1   
    num_frames = 100; %randi(100);
    Z = rand(num_frames, d);  
    Z = normr(Z); 
end

% thresh is \eta in the paper that controls the extend of temporal
% projection distance.
thresh = 0.001; num_subspaces=3; num_iter = 100; verbose = 1;
tic;
Y = msvmpool(seqs, Z, num_subspaces, thresh, num_iter, verbose);
toc
return;

