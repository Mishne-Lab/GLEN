clear;close all
addpath("../GLEN");
load('graphs/weighted_normalized_graphs_er03_n20_offset2-2_w2.mat');
% load('graphs/weighted_normalized_graphs_er02_n20_offset2-2_w2.mat');
% load('graphs/weighted_normalized_graphs_sbm0401_n20_offset2-2_w2.mat');
% load('graphs/weighted_normalized_graphs_ws201_n20_offset2-2_w2.mat');
% load('graphs/weighted_normalized_graphs_rg07505_n20_offset2-2_wg.mat');
% load('graphs/weighted_normalized_graphs_rg0102_n20_offset2-2_wg.mat');
%% Get a generated graph
nreplicate = 20; % repeat the same experiment (based on different graphs)
N = 20;
M = 2000;
for ii = 1:nreplicate

%% Load the graph Laplacian and signals

L_0 = data{ii,2};
X = data{ii,3};
X_noisy = data{ii,4};
X_noisy = X_noisy(:, 1:M);

%% main loop GLEN
alpha_glen = 10.^[0:-0.2:-2];
beta_glen = 10.^[-0.5:-0.1:-2];

precision_poiss_glen = zeros(length(alpha_glen),length(beta_glen),20);
recall_poiss_glen = zeros(length(alpha_glen),length(beta_glen),20);
Fmeasure_poiss_glen = zeros(length(alpha_glen),length(beta_glen),20);
NMI_poiss_glen = zeros(length(alpha_glen),length(beta_glen),20);
num_of_edges_poiss_glen = zeros(length(alpha_glen),length(beta_glen),20);
rel_err_poiss_glen = zeros(length(alpha_glen),length(beta_glen),20);

for i = 1:length(alpha_glen)
    for j = 1:length(beta_glen)
        param = struct();
        param.reg_type = 'cgl';
        param.max_iter = 20;
        param.alpha = alpha_glen(i);
        param.beta = beta_glen(j);
        param.gamma = 0;
        param.vi = 0; % param.vi = 0.5;

        [L,Y,offset,L_iter,O_iter,Y_iter] = glen_poisson(X_noisy, param);
        Ocell_poiss_glen{i,j} = O_iter;
        for k = 1:20
            if k <= size(L_iter,3)
                L = L_iter(:,:,k);
                L(abs(L)<10^(-4))=0;
                L = L - diag(sum(L,1));
                if all(L==0, 'all')
                    L = eye(N)-1/N;
                end
                L = L/trace(L)*N;
                Lcell_poiss_glen{i,j,k} = L;
                [precision_poiss_glen(i,j,k),recall_poiss_glen(i,j,k),Fmeasure_poiss_glen(i,j,k),NMI_poiss_glen(i,j,k),num_of_edges_poiss_glen(i,j,k)] = graph_learning_perf_eval(L_0,L);
                rel_err_poiss_glen(i,j,k) = norm(L-L_0, 'fro') / norm(L_0, 'fro');

                deg = diag(L);
                deg_0 = diag(L_0);
                edge = squareform(-L+diag(deg));
                edge_0 = squareform(-L_0+diag(deg_0));
                rel_err_edge1_poiss_glen(i,j,k) = norm(edge-edge_0, 1) / norm(edge_0, 1);
                rel_err_edge2_poiss_glen(i,j,k) = norm(edge-edge_0, 2) / norm(edge_0, 2);
                rel_err_deg1_poiss_glen(i,j,k) = norm(deg-deg_0, 1) / norm(deg_0, 1);
                rel_err_deg2_poiss_glen(i,j,k) = norm(deg-deg_0, 2) / norm(deg_0, 2);
                rel_err_y_poiss_glen(i,j,k) = norm(X-Y_iter(:,:,k), 'fro') / norm(X, 'fro');
            else
                Lcell_poiss_glen{i,j,k} = L;
                precision_poiss_glen(i,j,k) = precision_poiss_glen(i,j,size(L_iter,3));
                recall_poiss_glen(i,j,k) = recall_poiss_glen(i,j,size(L_iter,3));
                Fmeasure_poiss_glen(i,j,k) = Fmeasure_poiss_glen(i,j,size(L_iter,3));
                NMI_poiss_glen(i,j,k) = NMI_poiss_glen(i,j,size(L_iter,3));
                num_of_edges_poiss_glen(i,j,k) = num_of_edges_poiss_glen(i,j,size(L_iter,3));
                rel_err_poiss_glen(i,j,k) = rel_err_poiss_glen(i,j,size(L_iter,3));

                rel_err_edge1_poiss_glen(i,j,k) = rel_err_edge1_poiss_glen(i,j,size(L_iter,3));
                rel_err_edge2_poiss_glen(i,j,k) = rel_err_edge2_poiss_glen(i,j,size(L_iter,3));
                rel_err_deg1_poiss_glen(i,j,k) = rel_err_deg1_poiss_glen(i,j,size(L_iter,3));
                rel_err_deg2_poiss_glen(i,j,k) = rel_err_deg2_poiss_glen(i,j,size(L_iter,3));
                rel_err_y_poiss_glen(i,j,k) = rel_err_y_poiss_glen(i,j,size(L_iter,3));
            end
        end
    end
end

result1_poiss_glen(:,:,:,ii) = precision_poiss_glen;
result2_poiss_glen(:,:,:,ii) = recall_poiss_glen;
result3_poiss_glen(:,:,:,ii) = Fmeasure_poiss_glen;
result4_poiss_glen(:,:,:,ii) = NMI_poiss_glen;
result5_poiss_glen(:,:,:,ii) = num_of_edges_poiss_glen;
result6_poiss_glen(:,:,:,ii) = rel_err_poiss_glen;

result7_poiss_glen(:,:,:,ii) = rel_err_edge1_poiss_glen;
result8_poiss_glen(:,:,:,ii) = rel_err_edge2_poiss_glen;
result9_poiss_glen(:,:,:,ii) = rel_err_deg1_poiss_glen;
result10_poiss_glen(:,:,:,ii) = rel_err_deg2_poiss_glen;
result11_poiss_glen(:,:,:,ii) = rel_err_y_poiss_glen;

graph_poiss_glen{ii} = Lcell_poiss_glen;
offest_poiss_glen{ii} = Ocell_poiss_glen;

end