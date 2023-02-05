clear;close all
% load('graphs_er03_n20_nooffset.mat');
% load('graphs_er03_n20_offset2-2.mat');
% load('graphs/weighted_normalized_graphs_er03_n20_m10_offset2-2_w2.mat');
% load('graphs/weighted_normalized_graphs_sbm0401_n20_m10_offset2-2_w2.mat');
load('graphs/weighted_normalized_graphs_ws201_n20_m10_offset2-2_w2.mat');
% load('graphs_er03_n20_idlink_offset5.mat');
%% Generate a graph
nreplicate = 20; % repeat the same experiment (based on different graphs)
N = 20;
T = 2000;
for ii = 1:nreplicate

%% Load the graph Laplacian and signals

L_0 = data{ii,2};
X = data{ii,3};
X_noisy = data{ii,4};
X_noisy = X_noisy(:, 1:T, :);

%% main loop

%% lake
% lambda = 10.^[1:-0.2:-1];
% 
% % 40 nodes
% lambda = 10.^[0:-0.2:-2];
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
%     [precision_poiss_logdet(k),recall_poiss_logdet(k),Fmeasure_poiss_logdet(k),NMI_poiss_logdet(k),num_of_edges_poiss_logdet(k)] = graph_learning_perf_eval(L_0,L);
%     rel_err_poiss_logdet(k) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
% % end
% end
% 
% result1_poiss_logdet(:,ii) = precision_poiss_logdet;
% result2_poiss_logdet(:,ii) = recall_poiss_logdet;
% result3_poiss_logdet(:,ii) = Fmeasure_poiss_logdet;
% result4_poiss_logdet(:,ii) = NMI_poiss_logdet;
% result5_poiss_logdet(:,ii) = num_of_edges_poiss_logdet;
% result6_poiss_logdet(:,ii) = rel_err_poiss_logdet;
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
%     [precision_poiss_l2(j),recall_poiss_l2(j),Fmeasure_poiss_l2(j),NMI_poiss_l2(j),num_of_edges_poiss_l2(j)] = graph_learning_perf_eval(L_0,L);
%     rel_err_poiss_l2(j) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
% end
% 
% result1_poiss_l2(:,ii) = precision_poiss_l2;
% result2_poiss_l2(:,ii) = recall_poiss_l2;
% result3_poiss_l2(:,ii) = Fmeasure_poiss_l2;
% result4_poiss_l2(:,ii) = NMI_poiss_l2;
% result5_poiss_l2(:,ii) = num_of_edges_poiss_l2;
% result6_poiss_l2(:,ii) = rel_err_poiss_l2;
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
% % 40 nodes weighted, normalized, offset = 2, -2
% beta_log = 10.^[0:-0.1:-1];
% gamma_log = 10.^[-0.5:-0.1:-1.5];
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
%         [precision_poiss_log(j,k),recall_poiss_log(j,k),Fmeasure_poiss_log(j,k),NMI_poiss_log(j,k),num_of_edges_poiss_log(j,k)] = graph_learning_perf_eval(L_0,L);
%         rel_err_poiss_log(j,k) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
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
% graph_poiss_log{ii} = Lcell_poiss_log;

%% cgl-ours

% er no offset
beta_cgl = 10.^[-1:-0.1:-2]; % design 1
beta_cgl = 10.^[-0.7:-0.1:-1.7]; % design2
beta_cgl = 10.^[-1.7:-0.1:-2.7]; % log

% er offset = -1
beta_cgl = 10.^[-1:-0.1:-2]; % design 1
beta_cgl = 10.^[-2.7:-0.1:-3.7]; % log

% er offset = 2, -2
beta_cgl = 10.^[-1:-0.1:-2]; % design 1

% sbm offset = 2, -2
beta_cgl = 10.^[-0.5:-0.1:-1.5]; % design 1

% ws offset = 2, -2
beta_cgl = 10.^[-0.5:-0.1:-1.5]; % design 1

% ws 40 nodes offset = 2, -2
beta_cgl = 10.^[-0.5:-0.1:-1.5]; % design 1
alpha_cgl = 10.^[1:-0.2:-1];
% alpha_cgl = 10.^[2:-0.2:0];

% % er offset = 2, -2
% beta_cgl = 10.^[-1:-0.1:-2]; % design 1
% % beta_cgl = 10.^[-1.5:-0.1:-2.5]; % design 1
% alpha_cgl = 10.^[0:-0.2:-2];
% 
% % sbm offset = 2, -2
% beta_cgl = 10.^[-0.5:-0.1:-1.5]; % design 1
% beta_cgl = 10.^[-1:-0.1:-2]; % design 1
% alpha_cgl = 10.^[0:-0.2:-2];
% alpha_cgl = 10.^[0:-0.1:-1];

% % test init
% beta_cgl = 10.^[-1:-0.1:-2]; % design 1
% alpha_cgl = 10.^[0:-0.2:-1];

precision_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),20);
recall_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),20);
Fmeasure_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),20);
NMI_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),20);
num_of_edges_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),20);
rel_err_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),20);

length_alpha = length(alpha_cgl);
length_beta = length(beta_cgl);

% mu = mean(X_noisy,2);
% sigma = cov(X_noisy');
% S = log(((sigma-diag(mu))./mu)./mu'+1);
% A_mask = ones(size(S))-eye(size(S));
% S = cov(X');

for i = 1:length_alpha%1%
    for j = 1:length_beta
        beta = beta_cgl(j);
        % GL-SigRep
%         L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,40,1);
        param = struct();
        param.reg_type = 'cgl';
        param.max_iter = 20;
        param.alpha = alpha_cgl(i);
        param.beta = beta_cgl(j);
        param.gamma = 0;
%         param.L_init = L;
%         param.L_init = L_0;
%         param.Y_init = X;
%         offset = [2*ones(N/2,1);-2*ones(N/2,1)];
%         Y_init = zeros(N,T);
%         for t = 1:T
%             x = X_noisy(:,t);
%             x1 = x;
%             x1(x==0) = 1;
%             y = (diag(x) + L)\(-1 + x.*log(x1));
%             Y_init(:,t) = y;
%         end
%         Y_init = Y_init - mean(Y_init,1);
% %         Y_init = Y_init - offset;
%         param.Y_init = Y_init;
%         param.m = 5;
%         [L,Y,offset,L_iter] = gl_poisson_log(X_noisy, param);
        [L,Y,offset,L_iter,O_iter] = vgl_prod_poisson_log(X_noisy, param);
        Ycell_poiss_cgl1{i,j} = Y;
        Ocell_poiss_cgl1{i,j} = O_iter;
        for k = 1:20
            if k <= size(L_iter,3)
                L = L_iter(:,:,k);
                Lcell_poiss_cgl1{i,j,k} = L;
                L(abs(L)<10^(-4))=0;
                if any(isnan(L))
                    continue
                end
                [precision_poiss_cgl1(i,j,k),recall_poiss_cgl1(i,j,k),Fmeasure_poiss_cgl1(i,j,k),NMI_poiss_cgl1(i,j,k),num_of_edges_poiss_cgl1(i,j,k)] = graph_learning_perf_eval(L_0,L);
                rel_err_poiss_cgl1(i,j,k) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
            else
                precision_poiss_cgl1(i,j,k) = precision_poiss_cgl1(i,j,size(L_iter,3));
                recall_poiss_cgl1(i,j,k) = recall_poiss_cgl1(i,j,size(L_iter,3));
                Fmeasure_poiss_cgl1(i,j,k) = Fmeasure_poiss_cgl1(i,j,size(L_iter,3));
                NMI_poiss_cgl1(i,j,k) = NMI_poiss_cgl1(i,j,size(L_iter,3));
                num_of_edges_poiss_cgl1(i,j,k) = num_of_edges_poiss_cgl1(i,j,size(L_iter,3));
                rel_err_poiss_cgl1(i,j,k) = rel_err_poiss_cgl1(i,j,size(L_iter,3));
            end
        end
    end
end

result1_poiss_cgl1(:,:,:,ii) = precision_poiss_cgl1;
result2_poiss_cgl1(:,:,:,ii) = recall_poiss_cgl1;
result3_poiss_cgl1(:,:,:,ii) = Fmeasure_poiss_cgl1;
result4_poiss_cgl1(:,:,:,ii) = NMI_poiss_cgl1;
result5_poiss_cgl1(:,:,:,ii) = num_of_edges_poiss_cgl1;
result6_poiss_cgl1(:,:,:,ii) = rel_err_poiss_cgl1;

graph_poiss_cgl1{ii} = Lcell_poiss_cgl1;
offest_poiss_cgl1{ii} = Ocell_poiss_cgl1;

end
% save("weighted_normalized_results_ws201_n40_t2000_offset2-2");