clear;close all
%% load
load('animals.mat');
X_noisy = data;

%% set parameters
param = struct();
param.reg_type = 'cgl';
param.N = length(names);
param.T = length(features);
param.max_iter = 20;

%%
% alpha = 10.^[-1:-0.2:-2];
% % beta = 10.^[1:-0.5:-1];
% beta = 10.^[1:-0.2:0];

alpha = 10.^[0:-0.2:-1];
% beta = 10.^[1:-0.5:-1];
beta = 10.^[0.5:-0.2:-0.5];

%% latest
alpha = 10^(-0.4);
beta = 10^(0.1);
beta = 10^(0.4);
% beta = 0.02;
% lambda = 10;
% trace = 10^1.5;

%% egilmez demo
clear 
clc
close all


my_eps_outer = 1e-4; my_eps_inner = 1e-6; max_cycles = 40;
scale = 1;
isNormalized = 0;

load('animals.mat');
% % compute correlation matrix
S = cov(data',1); 
% for binary data we add +1/3 to diagonals(suggested by Banerjee et al. ''Model Selection Through Sparse Maximum Likelihood Estimation for Multivariate Gaussian or Binary Data (2008)
S = S  + (1/3)*eye(size(S));
A_mask=ones(size(S)) - eye(size(S));
alpha = 0.02;
% alpha = 0.08;
% [Laplacian,~,convergence] = estimate_ggl(S,A_mask,alpha,my_eps_outer,my_eps_inner,max_cycles,2);
%[Laplacian,~,convergence] = estimate_ddgl(S,A_mask,alpha,my_eps_outer,my_eps_inner,max_cycles,2);
[Laplacian,~,convergence] = estimate_cgl(S,A_mask,alpha,my_eps_outer,my_eps_inner,max_cycles,2);

Laplacian(abs(Laplacian) < my_eps_outer) = 0;  % threshold 
L = Laplacian;
A = -L+diag(diag(L));
g = graph(A);
% plot(g,'LineWidth',g.Edges.Weight*30,'NodeColor',"#A2142F",'EdgeColor',"#A2142F",'Layout','circle','NodeLabel',names,'NodeFontSize',10);
draw_animal_graph(Laplacian,names);

%% main loop
for i = 1:length(alpha)
    for j = 1:length(beta)
%         for k = 1:length(trace)
            param.alpha = alpha(i);
            param.beta = beta(j);
            param.gamma = 0;
%             param.trace = trace(k);

            [L,Y,offset,L_iter,O_iter] = gl_bernoulli_log(X_noisy, param);
%             Lcell{i,j} = L;
%         end
    end
end

%% performance
for i = 1:length(alpha)
    for j = 1:length(beta)
        subplot(5,5,5*(i-1)+j);
        L = Lcell{i,j};
        L = L / trace(L) * 33;
        A = -L+diag(diag(L));
        g = graph(A);
        plot(g,'LineWidth',g.Edges.Weight*10,'NodeColor',"#A2142F",'EdgeColor',"#A2142F",'Layout','circle','NodeLabel',names,'NodeFontSize',2);
%         draw_animal_graph(Lcell{i,j}, names);

    end
end
% draw_animal_graph(Lcell{5,3}, names);