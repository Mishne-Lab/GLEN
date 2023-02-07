%% Load the graph and signals
clear;close all
addpath("../GLEN");
load('area2_bump_train_spikes_resampled.mat');
load('area2_bump_trial_info.mat');
%% 65 neurons
Acell = zeros(length(spikes),2080);
Ocell = zeros(length(spikes),65);
Ycell = zeros(size(spikes));
Dcell = zeros(length(spikes),65);
fuck = zeros(length(spikes),65);
convergence = ones(length(spikes),1);
for t = 1:length(spikes)
%%
X_noisy = squeeze(spikes(t,:,:))';
active_nodes = find(mean(X_noisy,2));
X_noisy = X_noisy(active_nodes,:);
% [~,order] = sort(mean(X_noisy,2),'descend');
%% set parameters
param.N = size(X_noisy,1);
param.T = size(X_noisy,2);
param.max_iter = 50;

% alpha = 10.^[-1.1:-0.2:-1.9];
% beta = 10.^[-1.6:-0.2:-2.4];
% gamma = 10.^[-1:-0.5:-3];
alpha = 0.04;
beta = 0.01;
gamma = 0.1;

%% main loop

for i = 1:length(alpha)
    for j = 1:length(beta)
        for k = 1:length(gamma)
            param.reg_type = 'cgl';
            param.alpha = alpha(i);
            param.beta = beta(j);
            param.gamma = gamma(k);


            [L,Y,offset,L_iter,O_iter] = glen_tv_poisson(X_noisy, param);
            if any(isnan(L), 'all')
                convergence(t) = 0;
                continue
            end
            

            A = -L+diag(diag(L));
            A_full = zeros(65);
            A_full(active_nodes,active_nodes) = A;
            Acell(t,:) = squareform(A_full);
            Ycell(t,:,active_nodes) = Y';
            Dcell(t,active_nodes) = diag(L);
            Ocell(t,active_nodes) = offset;
        end
    end
end
end