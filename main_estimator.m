clear;close all
% load('graphs_er03_n20_nooffset.mat');
% load('graphs_er03_n20_offset2-2.mat');
load('graphs/weighted_normalized_graphs_er03_n20_offset2-2_w2.mat');
% load('graphs/weighted_normalized_graphs_er02_n20_offset2-2_w2.mat');
% load('graphs/weighted_normalized_graphs_sbm0401_n20_offset2-2_w2.mat');
% load('graphs/weighted_normalized_graphs_ws201_n20_offset2-2_w2.mat');
% load('graphs/weighted_normalized_graphs_ba1_n20_offset2-2_w2.mat');
% load('graphs/weighted_normalized_graphs_rg07505_n20_offset2-2_wg.mat');
% load('graphs/weighted_normalized_graphs_rg0102_n20_offset2-2_wg.mat');
% load('graphs_er03_n20_idlink_offset5.mat');
%% Generate a graph
nreplicate = 20; % repeat the same experiment (based on different graphs)
T = 2000;
for ii = 1:nreplicate

%% Load the graph Laplacian and signals

L_0 = data{ii,2};
X = data{ii,3};
X_noisy = data{ii,4};
X_noisy = X_noisy(:, 1:T);
N = size(X_noisy,1);

%% main loop

%% lake
% lambda = 10.^[1:-0.2:-1];
% lambda = 10.^[3:-0.4:-1];
% lambda = 10.^[2:-0.4:-2];
% 
% % % 40 nodes
% % lambda = 10.^[0:-0.2:-2];
% 
% precision_poiss_logdet = zeros(length(lambda),1);
% recall_poiss_logdet = zeros(length(lambda),1);
% Fmeasure_poiss_logdet = zeros(length(lambda),1);
% NMI_poiss_logdet = zeros(length(lambda),1);
% num_of_edges_poiss_logdet = zeros(length(lambda),1);
% rel_err_poiss_logdet = zeros(length(lambda),1);
% 
% Y = log(X_noisy+1);
% Y = Y - mean(Y,2);
% for k = 1:length(lambda)
%     param = struct();
%     param.reg_type = 'lasso';
%     param.max_iter = 50;
%     param.lambda = lambda(k);
%     % GL-LogDet
%     L = graph_learning_logdet_reglap(Y,param);
%     Lcell_poiss_logdet{k} = L;
%     L(abs(L)<10^(-4))=0;
%     L = L - diag(sum(L,1));
%     L = L/trace(L)*N;
%     [precision_poiss_logdet(k),recall_poiss_logdet(k),Fmeasure_poiss_logdet(k),NMI_poiss_logdet(k),num_of_edges_poiss_logdet(k)] = graph_learning_perf_eval(L_0,L);
%     rel_err_poiss_logdet(k) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
%     
%     deg = diag(L);
%     deg_0 = diag(L_0);
%     edge = squareform(-L+diag(deg));
%     edge_0 = squareform(-L_0+diag(deg_0));
%     rel_err_edge1_poiss_logdet(k) = norm(edge-edge_0, 1) / norm(edge_0, 1);
%     rel_err_edge2_poiss_logdet(k) = norm(edge-edge_0, 2) / norm(edge_0, 2);
%     rel_err_deg1_poiss_logdet(k) = norm(deg-deg_0, 1) / norm(deg_0, 1);
%     rel_err_deg2_poiss_logdet(k) = norm(deg-deg_0, 2) / norm(deg_0, 2);
% end
% 
% result1_poiss_logdet(:,ii) = precision_poiss_logdet;
% result2_poiss_logdet(:,ii) = recall_poiss_logdet;
% result3_poiss_logdet(:,ii) = Fmeasure_poiss_logdet;
% result4_poiss_logdet(:,ii) = NMI_poiss_logdet;
% result5_poiss_logdet(:,ii) = num_of_edges_poiss_logdet;
% result6_poiss_logdet(:,ii) = rel_err_poiss_logdet;
% 
% result7_poiss_logdet(:,ii) = rel_err_edge1_poiss_logdet;
% result8_poiss_logdet(:,ii) = rel_err_edge2_poiss_logdet;
% result9_poiss_logdet(:,ii) = rel_err_deg1_poiss_logdet;
% result10_poiss_logdet(:,ii) = rel_err_deg2_poiss_logdet;
% 
% graph_poiss_logdet{ii} = Lcell_poiss_logdet;

%% dong

% % offset = 0
% beta_l2 = 10.^[-0.5:-0.1:-1.5];
% 
% % offset = -1
% beta_l2 = 10.^[-1.5:-0.1:-2.5];
% 
% % offset = 2, -2
% beta_l2 = 10.^[-1.5:-0.1:-2.5];
% 
% % weighted, normalized, offset = 2, -2
% beta_l2 = 10.^[-0:-0.1:-1];
% 
% precision_poiss_l2 = zeros(length(beta_l2),1);
% recall_poiss_l2 = zeros(length(beta_l2),1);
% Fmeasure_poiss_l2 = zeros(length(beta_l2),1);
% NMI_poiss_l2 = zeros(length(beta_l2),1);
% num_of_edges_poiss_l2 = zeros(length(beta_l2),1);
% rel_err_poiss_l2 = zeros(length(beta_l2),1);
% 
% length_beta = length(beta_l2);
% 
% Y = log(X_noisy+1);
% Y = Y - mean(Y,2);
% 
% % offset = mean(log(X_noisy+1),2);
% % offset = offset - 0.5*log((var(X_noisy,0,2)-mean(X_noisy,2))./mean(X_noisy,2).^2+1);
% % % offset = zeros(20,1);
% % Y = log(exp(log(X_noisy+1)+0.5*exp(offset)./(1+exp(offset)).^2)-1)-offset;
% 
% for j = 1:length_beta
%     beta = beta_l2(j);
%     % new
%     Z = gsp_distanz(Y').^2;
%     W = gsp_learn_graph_l2_degrees(Z/beta/4/T,1,struct("maxit",200));
%     W(W<0) = 0;
%     L = diag(sum(W,1))-W;
%     Lcell_poiss_l2{j} = L;
%     L(abs(L)<10^(-4))=0;
%     L = L - diag(sum(L,1));
%     L = L/trace(L)*N;
%     [precision_poiss_l2(j),recall_poiss_l2(j),Fmeasure_poiss_l2(j),NMI_poiss_l2(j),num_of_edges_poiss_l2(j)] = graph_learning_perf_eval(L_0,L);
%     rel_err_poiss_l2(j) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
% 
%     deg = diag(L);
%     deg_0 = diag(L_0);
%     edge = squareform(-L+diag(deg));
%     edge_0 = squareform(-L_0+diag(deg_0));
%     rel_err_edge1_poiss_l2(j) = norm(edge-edge_0, 1) / norm(edge_0, 1);
%     rel_err_edge2_poiss_l2(j) = norm(edge-edge_0, 2) / norm(edge_0, 2);
%     rel_err_deg1_poiss_l2(j) = norm(deg-deg_0, 1) / norm(deg_0, 1);
%     rel_err_deg2_poiss_l2(j) = norm(deg-deg_0, 2) / norm(deg_0, 2);
% end
% 
% result1_poiss_l2(:,ii) = precision_poiss_l2;
% result2_poiss_l2(:,ii) = recall_poiss_l2;
% result3_poiss_l2(:,ii) = Fmeasure_poiss_l2;
% result4_poiss_l2(:,ii) = NMI_poiss_l2;
% result5_poiss_l2(:,ii) = num_of_edges_poiss_l2;
% result6_poiss_l2(:,ii) = rel_err_poiss_l2;
% 
% result7_poiss_l2(:,ii) = rel_err_edge1_poiss_l2;
% result8_poiss_l2(:,ii) = rel_err_edge2_poiss_l2;
% result9_poiss_l2(:,ii) = rel_err_deg1_poiss_l2;
% result10_poiss_l2(:,ii) = rel_err_deg2_poiss_l2;
% 
% graph_poiss_l2{ii} = Lcell_poiss_l2;

%% kalofolias
% % no offset
% beta_log = 10.^[0.5:-0.1:-0.5];
% gamma_log = 10.^[0:-0.1:-1];
% 
% % offset = -1
% beta_log = 10.^[-0.5:-0.1:-1.5];
% gamma_log = 10.^[-1:-0.1:-2];
% 
% % offset = 2, -2
% beta_log = 10.^[-0.5:-0.1:-1.5];
% gamma_log = 10.^[-1:-0.1:-2];
% 
% % weighted, normalized, offset = 2, -2
% beta_log = 10.^[0.5:-0.1:-0.5];
% gamma_log = 10.^[-0:-0.1:-1];
% 
% beta_log = 10.^[1:-0.2:-1];
% gamma_log = 10.^[0.5:-0.2:-1.5];
% 
% % test second highend
% beta_log = 10.^[1:-0.2:-1];
% gamma_log = 10.^[0:-0.2:-2];
% 
% % % 40 nodes weighted, normalized, offset = 2, -2
% % beta_log = 10.^[0:-0.1:-1];
% % gamma_log = 10.^[-0.5:-0.1:-1.5];
% 
% precision_poiss_log = zeros(length(beta_log),length(gamma_log));
% recall_poiss_log = zeros(length(beta_log),length(gamma_log));
% Fmeasure_poiss_log = zeros(length(beta_log),length(gamma_log));
% NMI_poiss_log = zeros(length(beta_log),length(gamma_log));
% num_of_edges_poiss_log = zeros(length(beta_log),length(gamma_log));
% 
% precision_kalofolias_poiss = zeros(length(beta_log),length(gamma_log));
% recall_kalofolias_poiss = zeros(length(beta_log),length(gamma_log));
% Fmeasure_kalofolias_poiss = zeros(length(beta_log),length(gamma_log));
% NMI_kalofolias_poiss = zeros(length(beta_log),length(gamma_log));
% num_of_edges_kalofolias_poiss = zeros(length(beta_log),length(gamma_log));
% 
% length_beta = length(beta_log);
% length_gamma = length(gamma_log);
% 
% Y = log(X_noisy+1);
% Y = Y - mean(Y,2);
% for j = 1:length_beta
%     for k = 1:length_gamma
%         beta = beta_log(j);
%         gamma = gamma_log(k);
%         % GL-SigRep
%         Z = gsp_distanz(Y').^2;
%         W = gsp_learn_graph_log_degrees(Z/beta/2/T,1,gamma/beta/2);
%         W(W<0) = 0;
%         L = diag(sum(W,1))-W;
%         Lcell_poiss_log{j,k} = L;
%         L(abs(L)<10^(-4))=0;
%         L = L - diag(sum(L,1));
%         L = L/trace(L)*N;
%         [precision_poiss_log(j,k),recall_poiss_log(j,k),Fmeasure_poiss_log(j,k),NMI_poiss_log(j,k),num_of_edges_poiss_log(j,k)] = graph_learning_perf_eval(L_0,L);
%         rel_err_poiss_log(j,k) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
% 
%         deg = diag(L);
%         deg_0 = diag(L_0);
%         edge = squareform(-L+diag(deg));
%         edge_0 = squareform(-L_0+diag(deg_0));
%         rel_err_edge1_poiss_log(j,k) = norm(edge-edge_0, 1) / norm(edge_0, 1);
%         rel_err_edge2_poiss_log(j,k) = norm(edge-edge_0, 2) / norm(edge_0, 2);
%         rel_err_deg1_poiss_log(j,k) = norm(deg-deg_0, 1) / norm(deg_0, 1);
%         rel_err_deg2_poiss_log(j,k) = norm(deg-deg_0, 2) / norm(deg_0, 2);
%     end
% end
% 
% result1_poiss_log(:,:,ii) = precision_poiss_log;
% result2_poiss_log(:,:,ii) = recall_poiss_log;
% result3_poiss_log(:,:,ii) = Fmeasure_poiss_log;
% result4_poiss_log(:,:,ii) = NMI_poiss_log;
% result5_poiss_log(:,:,ii) = num_of_edges_poiss_log;
% result6_poiss_log(:,:,ii) = rel_err_poiss_log;
% 
% result7_poiss_log(:,:,ii) = rel_err_edge1_poiss_log;
% result8_poiss_log(:,:,ii) = rel_err_edge2_poiss_log;
% result9_poiss_log(:,:,ii) = rel_err_deg1_poiss_log;
% result10_poiss_log(:,:,ii) = rel_err_deg2_poiss_log;
% 
% graph_poiss_log{ii} = Lcell_poiss_log;

%% cgl

% er no offset
beta_cgl = 10.^[-1:-0.1:-2]; % design 1
beta_cgl = 10.^[-0.7:-0.1:-1.7]; % design2
beta_cgl = 10.^[-1.7:-0.1:-2.7]; % log

% er offset = -1
beta_cgl = 10.^[-1:-0.1:-2]; % design 1
beta_cgl = 10.^[-2.7:-0.1:-3.7]; % log

% er offset = 2, -2
beta_cgl = 10.^[-1:-0.1:-2]; % design 1
beta_cgl = 10.^[-2.7:-0.1:-3.7]; % log

% er03 weighted, normalized, offset = 2, -2!
beta_cgl = 10.^[-1:-0.1:-2]; % design 1
beta_cgl = 10.^[-2.5:-0.1:-3.5]; % log

% % % er02 weighted, normalized, offset = 2, -2!
% % beta_cgl = 10.^[-2:-0.1:-3]; % log

% % sbm offset = 2, -2!
% beta_cgl = 10.^[-2.5:-0.1:-3.5]; % log
% % beta_cgl = 10.^[-2.5:-0.2:-4.5]; % log
% 
% % ws offset = 2, -2!
% beta_cgl = 10.^[-2:-0.1:-3]; % log

% % rg-rw offset = 2, -2
% beta_cgl = 10.^[-2:-0.1:-3]; % log

% % rg offset = 2, -2
% beta_cgl = 10.^[-2.5:-0.1:-3.5]; % log

% % ba offset = 2, -2
% beta_cgl = 10.^[-1.5:-0.1:-2.5]; % log

% % ws 40 nodes offset = 2, -2
% beta_cgl = 10.^[-1.5:-0.1:-2.5]; % log

precision_poiss_cgl1 = zeros(length(beta_cgl),1);
recall_poiss_cgl1 = zeros(length(beta_cgl),1);
Fmeasure_poiss_cgl1 = zeros(length(beta_cgl),1);
NMI_poiss_cgl1 = zeros(length(beta_cgl),1);
num_of_edges_poiss_cgl1 = zeros(length(beta_cgl),1);
rel_err_poiss_cgl1 = zeros(length(beta_cgl),1);

length_beta = length(beta_cgl);

mu = mean(X_noisy,2);
sigma = cov(X_noisy');
S = log((sigma./mu)./mu'+1);
S2 = log(((sigma-diag(mu))./mu)./mu'+1);
S = S + diag(log((var(X_noisy,0,2)-mean(X_noisy,2))./mean(X_noisy,2).^2+1)-diag(S));
% S = 0.8*S + 0.2*trace(S)/10*eye(size(S));
% [V,D] = eig(S);
% D(D<0) = 0;
% S = V * D * V';
A_mask = ones(size(S))-eye(size(S));

S = cov(log(X_noisy+1)');

for j = 1:length_beta%10%5:15%5:11%
    beta = beta_cgl(j);
    % GL-SigRep
    L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,40,1);
    Lcell_poiss_cgl1{j} = L;
    L(abs(L)<10^(-4))=0;
    L = L - diag(sum(L,1));
    L = L/trace(L)*N;
    [precision_poiss_cgl1(j),recall_poiss_cgl1(j),Fmeasure_poiss_cgl1(j),NMI_poiss_cgl1(j),num_of_edges_poiss_cgl1(j)] = graph_learning_perf_eval(L_0,L);
    rel_err_poiss_cgl1(j) = norm(L-L_0, 'fro') / norm(L_0, 'fro');

    deg = diag(L);
    deg_0 = diag(L_0);
    edge = squareform(-L+diag(deg));
    edge_0 = squareform(-L_0+diag(deg_0));
    rel_err_edge1_poiss_cgl1(j) = norm(edge-edge_0, 1) / norm(edge_0, 1);
    rel_err_edge2_poiss_cgl1(j) = norm(edge-edge_0, 2) / norm(edge_0, 2);
    rel_err_deg1_poiss_cgl1(j) = norm(deg-deg_0, 1) / norm(deg_0, 1);
    rel_err_deg2_poiss_cgl1(j) = norm(deg-deg_0, 2) / norm(deg_0, 2);
end

result1_poiss_cgl1(:,ii) = precision_poiss_cgl1;
result2_poiss_cgl1(:,ii) = recall_poiss_cgl1;
result3_poiss_cgl1(:,ii) = Fmeasure_poiss_cgl1;
result4_poiss_cgl1(:,ii) = NMI_poiss_cgl1;
result5_poiss_cgl1(:,ii) = num_of_edges_poiss_cgl1;
result6_poiss_cgl1(:,ii) = rel_err_poiss_cgl1;

result7_poiss_cgl1(:,ii) = rel_err_edge1_poiss_cgl1;
result8_poiss_cgl1(:,ii) = rel_err_edge2_poiss_cgl1;
result9_poiss_cgl1(:,ii) = rel_err_deg1_poiss_cgl1;
result10_poiss_cgl1(:,ii) = rel_err_deg2_poiss_cgl1;

graph_poiss_cgl1{ii} = Lcell_poiss_cgl1;

%% cgl-ours
% 
% % er no offset
% beta_cgl = 10.^[-1:-0.1:-2]; % design 1
% beta_cgl = 10.^[-0.7:-0.1:-1.7]; % design2
% beta_cgl = 10.^[-1.7:-0.1:-2.7]; % log
% 
% % er offset = -1
% beta_cgl = 10.^[-1:-0.1:-2]; % design 1
% beta_cgl = 10.^[-2.7:-0.1:-3.7]; % log
% 
% % er03 offset = 2, -2
% beta_cgl = 10.^[-1:-0.1:-2]; % design 1
% beta_cgl = 10.^[-1:-0.3:-4]; % design 1
% 
% % % er02 offset = 2, -2
% % beta_cgl = 10.^[-0.6:-0.1:-1.6]; % design 1
% % 
% % sbm offset = 2, -2
% beta_cgl = 10.^[-0.5:-0.1:-1.5]; % design 1
% beta_cgl = 10.^[-0.8:-0.1:-1.8]; % design 1
% 
% % ws offset = 2, -2
% beta_cgl = 10.^[-0.5:-0.1:-1.5]; % design 1
% 
% % % rg-rw offset = 2, -2
% % beta_cgl = 10.^[-1:-0.1:-2]; % design 1
% 
% % % rg offset = 2, -2
% % beta_cgl = 10.^[-0.9:-0.1:-1.9]; % design 1
% 
% % % ba offset = 2, -2
% % beta_cgl = 10.^[-0.3:-0.1:-1.3]; % design 1
% 
% % % ws 40 nodes offset = 2, -2
% % beta_cgl = 10.^[-0:-0.1:-1]; % design 1
% 
% precision_poiss_cgl1_new = zeros(length(beta_cgl),1);
% recall_poiss_cgl1_new = zeros(length(beta_cgl),1);
% Fmeasure_poiss_cgl1_new = zeros(length(beta_cgl),1);
% NMI_poiss_cgl1_new = zeros(length(beta_cgl),1);
% num_of_edges_poiss_cgl1_new = zeros(length(beta_cgl),1);
% rel_err_poiss_cgl1_new = zeros(length(beta_cgl),1);
% 
% length_beta = length(beta_cgl);
% 
% mu = mean(X_noisy,2);
% sigma = cov(X_noisy');
% S = log(((sigma-diag(mu))./mu)./mu'+1);
% A_mask = ones(size(S))-eye(size(S));
% 
% for j = 1:length_beta%10%5:15%5:11%
%     beta = beta_cgl(j);
%     % GL-SigRep
%     L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,40,1);
%     Lcell_poiss_cgl1_new{j} = L;
%     if any(isnan(L), 'all')
%         L = eye(N)-1/N;
%     end
%     L(abs(L)<10^(-4))=0;
%     L = L - diag(sum(L,1));
%     L = L/trace(L)*N;
%     [precision_poiss_cgl1_new(j),recall_poiss_cgl1_new(j),Fmeasure_poiss_cgl1_new(j),NMI_poiss_cgl1_new(j),num_of_edges_poiss_cgl1_new(j)] = graph_learning_perf_eval(L_0,L);
%     rel_err_poiss_cgl1_new(j) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
% 
%     deg = diag(L);
%     deg_0 = diag(L_0);
%     edge = squareform(-L+diag(deg));
%     edge_0 = squareform(-L_0+diag(deg_0));
%     rel_err_edge1_poiss_cgl1_new(j) = norm(edge-edge_0, 1) / norm(edge_0, 1);
%     rel_err_edge2_poiss_cgl1_new(j) = norm(edge-edge_0, 2) / norm(edge_0, 2);
%     rel_err_deg1_poiss_cgl1_new(j) = norm(deg-deg_0, 1) / norm(deg_0, 1);
%     rel_err_deg2_poiss_cgl1_new(j) = norm(deg-deg_0, 2) / norm(deg_0, 2);
% end
% 
% result1_poiss_cgl1_new(:,ii) = precision_poiss_cgl1_new;
% result2_poiss_cgl1_new(:,ii) = recall_poiss_cgl1_new;
% result3_poiss_cgl1_new(:,ii) = Fmeasure_poiss_cgl1_new;
% result4_poiss_cgl1_new(:,ii) = NMI_poiss_cgl1_new;
% result5_poiss_cgl1_new(:,ii) = num_of_edges_poiss_cgl1_new;
% result6_poiss_cgl1_new(:,ii) = rel_err_poiss_cgl1_new;
% 
% result7_poiss_cgl1_new(:,ii) = rel_err_edge1_poiss_cgl1_new;
% result8_poiss_cgl1_new(:,ii) = rel_err_edge2_poiss_cgl1_new;
% result9_poiss_cgl1_new(:,ii) = rel_err_deg1_poiss_cgl1_new;
% result10_poiss_cgl1_new(:,ii) = rel_err_deg2_poiss_cgl1_new;
% 
% graph_poiss_cgl1_new{ii} = Lcell_poiss_cgl1_new;

end
% save("./results/weighted_normalized_results_5it_er03_n20_t2000_offset2-2");