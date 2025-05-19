clear;close all
load('example_binomial_graph_signals.mat')

%% Get a generated graph
M = num_of_signal;
for ii = 1:nreplicate

%% Load the graph Laplacian and signals

L_0 = data{ii,2};
X = data{ii,3};
X_noisy = data{ii,4};
X_noisy = X_noisy(:, 1:M);

%% main loop GLEN

beta_glen = [0.01,0.02,0.03,0.04,0.05];
gamma_glen = [0.4,0.5,0.6,0.7,0.8];

len_beta = length(beta_glen);
len_gamma = length(gamma_glen);

for i = 1:len_gamma
    for j = 1:len_beta
        param = struct();
        param.init = 'cgl';
        param.lsolver = 'gd';
        param.max_iter = 100;
        param.max_iter_inner = 1;
        param.alpha = 0;
        param.gamma = gamma_glen(i);
        param.beta = beta_glen(j);
        param.K = K;
        param.step_size = 0.001;
        param.tol = 1e-4;

        [L,Y,llp] = glen_binomial(X_noisy, param);
        Lcell_bino_glen{j,i} = L;
    end
end

graph_bino_glen{ii} = Lcell_bino_glen;

end