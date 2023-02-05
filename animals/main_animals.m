%% GLEN demo
clear;close all
%% load
load('animals.mat');
X_noisy = data;
N = length(names);
T = length(features);

%% set-up parameters
param = struct();
param.reg_type = 'cgl';
param.max_iter = 20;

% alpha = 10.^[0:-0.2:-1]; % grid search
% beta = 10.^[0.5:-0.2:-0.5];

alpha = 10^(-0.4);
beta = 10^(0.5);

%% run GLEN-Bernoulli
for i = 1:length(alpha)
    for j = 1:length(beta)
            param.alpha = alpha(i);
            param.beta = beta(j);
            param.gamma = 0;

            [L,Y,offset,L_iter,O_iter] = gl_bernoulli_log(X_noisy, param);
            Lcell{i,j} = L;
    end
end

%% visualize
gplot_animals(L, names);
