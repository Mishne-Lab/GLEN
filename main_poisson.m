clear;close all
% load('graphs_er03_n20_nooffset.mat');
load('graphs_er03_n20_idlink_offset5.mat');
%% Generate a graph
nreplicate = 10; % repeat the same experiment (based on different graphs)
T = 200;
for ii = 1:nreplicate

%% Load the graph Laplacian and signals

L_0 = data{ii,2};
X = data{ii,3};
X_noisy = data{ii,4};
X_noisy = X_noisy(:, 1:T);

%% main loop

%% lake
% lambda = 10.^[2:-0.1:0];
% for k = 1:length(lambda)
%     param = struct();
%     param.reg_type = 'lasso';
%     param.max_iter = 50;
%     param.lambda = lambda(k);
%     param.beta = 0;
%     param.gamma = 0;
%     param.rho = 0;
%     % GL-LogDet
%     [L_lake,Y,L_poiss_logdet,offset] = vgl_poisson_fast(X_noisy,param);
%     Lcell_lake{k} = L_lake;
%     Lcell_poiss_logdet{k} = L_poiss_logdet;
%     L_poiss_logdet(abs(L_poiss_logdet)<10^(-4))=0;
%     [precision_lake(k),recall_lake(k),Fmeasure_lake(k),NMI_lake(k),num_of_edges_lake(k)] = graph_learning_perf_eval(L_0,L_lake);
%     L_poiss_logdet(abs(L_poiss_logdet)<10^(-4))=0;
%     [precision_poiss_logdet(k),recall_poiss_logdet(k),Fmeasure_poiss_logdet(k),NMI_poiss_logdet(k),num_of_edges_poiss_logdet(k)] = graph_learning_perf_eval(L_0,L_poiss_logdet);
% end
%% dong
alpha_l2 = 10.^[0:-0.2:-2]; % t=200
beta_l2 = 10.^[0:-0.1:-1];

alpha_l2 = 10.^[1:-0.2:-1];
beta_l2 = 10.^[0.5:-0.1:-0.5]; % identity link

precision_poiss_l2 = zeros(length(alpha_l2),length(beta_l2),10);
recall_poiss_l2 = zeros(length(alpha_l2),length(beta_l2),10);
Fmeasure_poiss_l2 = zeros(length(alpha_l2),length(beta_l2),10);
NMI_poiss_l2 = zeros(length(alpha_l2),length(beta_l2),10);
num_of_edges_poiss_l2 = zeros(length(alpha_l2),length(beta_l2),10);

precision_harvard_poiss = zeros(length(alpha_l2),length(beta_l2),10);
recall_harvard_poiss = zeros(length(alpha_l2),length(beta_l2),10);
Fmeasure_harvard_poiss = zeros(length(alpha_l2),length(beta_l2),10);
NMI_harvard_poiss = zeros(length(alpha_l2),length(beta_l2),10);
num_of_edges_harvard_poiss = zeros(length(alpha_l2),length(beta_l2),10);

length_alpha = length(alpha_l2);
length_beta = length(beta_l2);

% parfor (i = 1:length_alpha, 7)%3:13%
for i = 1:length_alpha
    for j = 1:length_beta%8:13%
        param = struct();
        param.reg_type = 'l2';
        param.max_iter = 50;
        param.damped = 0;
        param.alpha = alpha_l2(i);
        param.beta = beta_l2(j);
        param.gamma = 0;
        param.rho = 10;
        % new
%         [L,Y,L_harvard,offset,L_iter] = gl_poisson_log(X_noisy,param);
        [L,Y,L_harvard,offset,L_iter] = gl_poisson_identity_interior(X_noisy,param);
        Ycell_poiss_l2{i,j} = Y;
        for k = 1:size(L_iter,3)
            L = L_iter(:,:,k);
            Lcell_poiss_l2{i,j,k} = L;
            L(abs(L)<10^(-4))=0;
            if any(isnan(L))
                continue
            end
            [precision_poiss_l2(i,j,k),recall_poiss_l2(i,j,k),Fmeasure_poiss_l2(i,j,k),NMI_poiss_l2(i,j,k),num_of_edges_poiss_l2(i,j,k)] = graph_learning_perf_eval(L_0,L);
        end

%         Lcell_poiss_l2{i,j} = L;
%         Lcell_harvard_poiss{i,j} = L_harvard;
%         Ycell_poiss_l2{i,j} = Y;
%         Ocell_poiss_l2{i,j} = offset;
%         L(abs(L)<10^(-4))=0;
%         [precision_poiss_l2(i,j),recall_poiss_l2(i,j),Fmeasure_poiss_l2(i,j),NMI_poiss_l2(i,j),num_of_edges_poiss_l2(i,j)] = graph_learning_perf_eval(L_0,L);
%         L_harvard(abs(L_harvard)<10^(-4))=0;
%         [precision_harvard_poiss(i,j),recall_harvard_poiss(i,j),Fmeasure_harvard_poiss(i,j),NMI_harvard_poiss(i,j),num_of_edges_harvard_poiss(i,j)] = graph_learning_perf_eval(L_0,L_harvard);
    end
end

%% kalofolias
% % er02
% alpha_log = 10.^[4:-0.5:1];
% beta_log = 10.^[0.5:-0.1:-0.5];
% gamma_log = 10.^[3:-0.5:-1];
% 
% precision_poiss_log = zeros(length(alpha_log),length(beta_log),length(gamma_log));
% recall_poiss_log = zeros(length(alpha_log),length(beta_log),length(gamma_log));
% Fmeasure_poiss_log = zeros(length(alpha_log),length(beta_log),length(gamma_log));
% NMI_poiss_log = zeros(length(alpha_log),length(beta_log),length(gamma_log));
% num_of_edges_poiss_log = zeros(length(alpha_log),length(beta_log),length(gamma_log));
% 
% precision_kalofolias_poiss = zeros(length(alpha_log),length(beta_log),length(gamma_log));
% recall_kalofolias_poiss = zeros(length(alpha_log),length(beta_log),length(gamma_log));
% Fmeasure_kalofolias_poiss = zeros(length(alpha_log),length(beta_log),length(gamma_log));
% NMI_kalofolias_poiss = zeros(length(alpha_log),length(beta_log),length(gamma_log));
% num_of_edges_kalofolias_poiss = zeros(length(alpha_log),length(beta_log),length(gamma_log));
% 
% length_alpha = length(alpha_log);
% length_beta = length(beta_log);
% length_gamma = length(gamma_log);
% 
% parfor (i = 1:length_alpha,7)%3:10%16:21%
%     for j = 1:length_beta%5:12%10:15%
%         for k = 1:length_gamma%3:7%
%             param = struct();
%             param.reg_type = 'log';
%             param.max_iter = 50;
%             param.damped = 0;
%             param.alpha = alpha_log(i);
%             param.beta = beta_log(j);
%             param.gamma = gamma_log(k);
%             % GL-SigRep
% %             [L,Y,L_kalofolias,offset] = graph_learning_poisson_new(X_noisy,param);
% %             Lcell_poiss_log{i,j,k} = L;
% %             Lcell_kalofolias_poiss{i,j,k} = L_kalofolias;
% %             Ycell_poiss_log{i,j,k} = Y;
% %             Ocell_poiss_log{i,j,k} = offset;
%             L(abs(L)<10^(-2))=0;
%             [precision_poiss_log(i,j,k),recall_poiss_log(i,j,k),Fmeasure_poiss_log(i,j,k),NMI_poiss_log(i,j,k),num_of_edges_poiss_log(i,j,k)] = graph_learning_perf_eval(L_0,L);
%             L_kalofolias(abs(L_kalofolias)<10^(-2))=0;
%             [precision_kalofolias_poiss(i,j,k),recall_kalofolias_poiss(i,j,k),Fmeasure_kalofolias_poiss(i,j,k),NMI_kalofolias_poiss(i,j,k),num_of_edges_kalofolias_poiss(i,j,k)] = graph_learning_perf_eval(L_0,L_kalofolias);
%         end
%     end
% end

% %% cgl
% % alpha = 10.^[0:-0.05:-1];
% % % beta = 10.^[2:-0.1:1]; % p=0.3
% % beta = 10.^[1.5:-0.1:-0.5]; % p=0.4
% 
% % alpha = 10.^[-0:-0.05:-1];
% % beta = 10.^[1.5:-0.1:-0.5];
% % 
% % alpha = 10.^[0:-0.1:-2];
% % beta = 10.^[1.25:-0.05:0.25];
% % 
% % alpha = 10.^[0:-0.05:-1];
% % beta = 10.^[2:-0.05:0.5];
% % beta = 10.^[3:-0.1:2]; % continue
% % 
% % % for sbm41
% % alpha = 10.^[-0.75:-0.05:-1.25];
% % beta = 10.^[2.75:-0.05:2.25];
% % 
% % % for er3
% % alpha = 10.^[-0.5:-0.025:-1].*2000;
% % beta = 10.^[2.5:-0.0125:2.25]./2000;
% % beta = 10.^[2.5:-0.05:2]./2000;
% 
% % er03 best
% % alpha_cgl = 10.^[0];
% alpha_cgl = 10.^[1:-0.2:-1];
% % beta_cgl = 10.^[-1.25:-0.05:-2.25];
% % beta_cgl = 10.^[2.5:-0.1:1]/2000;
% % beta_cgl = 10.^[2.2:-0.1:0.7]/2000;
% 
% beta_cgl = 10.^[-1:-0.1:-3];
% 
% % % er02
% % alpha_cgl = 10.^[4:-0.1:2];
% % beta_cgl = 10.^[2.5:-0.1:1]/2000;
% 
% precision_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl));
% recall_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl));
% Fmeasure_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl));
% NMI_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl));
% num_of_edges_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl));
% 
% precision_egilmez_poiss1 = zeros(length(alpha_cgl),length(beta_cgl));
% recall_egilmez_poiss1 = zeros(length(alpha_cgl),length(beta_cgl));
% Fmeasure_egilmez_poiss1 = zeros(length(alpha_cgl),length(beta_cgl));
% NMI_egilmez_poiss1 = zeros(length(alpha_cgl),length(beta_cgl));
% num_of_edges_egilmez_poiss1 = zeros(length(alpha_cgl),length(beta_cgl));
% 
% length_alpha = length(alpha_cgl);
% length_beta = length(beta_cgl);
% 
% % i = 1;
% % parfor (i = 1:length_alpha,7)%12%7:20%7:17%5:9%
% for i = 1:length_alpha
%     for j = 1:length_beta%10%5:15%5:11%
%         param = struct();
%         param.reg_type = 'cgl';
%         param.max_iter = 50;
%         param.damped = 0;
%         param.alpha = alpha_cgl(i);
%         param.beta = beta_cgl(j);
%         param.gamma = 0;
%         param.rho = 1;
%         % GL-SigRep
% %         [L,Y,L_egilmez,offset] = vgl_poisson_fast(X_noisy,param);
%         [L,Y,L_egilmez,offset] = gl_poisson_fast(X_noisy,param);
% %         [L,Y,L_egilmez,offset] = vgl_poisson_gd(X_noisy,param);
%         Lcell_poiss_cgl1{i,j} = L;
%         Lcell_egilmez_poiss1{i,j} = L_egilmez;
%         Ycell_poiss_cgl1{i,j} = Y;
%         Ocell_poiss_cgl1{i,j} = offset;
%         L(abs(L)<10^(-4))=0;
%         if any(isnan(L))
%             continue
%         end
%         [precision_poiss_cgl1(i,j),recall_poiss_cgl1(i,j),Fmeasure_poiss_cgl1(i,j),NMI_poiss_cgl1(i,j),num_of_edges_poiss_cgl1(i,j)] = graph_learning_perf_eval(L_0,L);
%         L_egilmez(abs(L_egilmez)<10^(-4))=0;
%         [precision_egilmez_poiss1(i,j),recall_egilmez_poiss1(i,j),Fmeasure_egilmez_poiss1(i,j),NMI_egilmez_poiss1(i,j),num_of_edges_egilmez_poiss1(i,j)] = graph_learning_perf_eval(L_0,L_egilmez);
%     end
% end

%% cgl

% er03 best
alpha_cgl = 10.^[1:-0.2:-1];
beta_cgl = 10.^[-1:-0.2:-3];


alpha_cgl = 10.^[1:-0.2:-1]; % identity & perturbed init
beta_cgl = 10.^[-0:-0.2:-2];

precision_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),10);
recall_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),10);
Fmeasure_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),10);
NMI_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),10);
num_of_edges_poiss_cgl1 = zeros(length(alpha_cgl),length(beta_cgl),10);

precision_egilmez_poiss1 = zeros(length(alpha_cgl),length(beta_cgl),10);
recall_egilmez_poiss1 = zeros(length(alpha_cgl),length(beta_cgl),10);
Fmeasure_egilmez_poiss1 = zeros(length(alpha_cgl),length(beta_cgl),10);
NMI_egilmez_poiss1 = zeros(length(alpha_cgl),length(beta_cgl),10);
num_of_edges_egilmez_poiss1 = zeros(length(alpha_cgl),length(beta_cgl),10);

length_alpha = length(alpha_cgl);
length_beta = length(beta_cgl);

% i = 1;
% parfor (i = 1:length_alpha,7)%12%7:20%7:17%5:9%
for i = 1:length_alpha
    for j = 1:length_beta%10%5:15%5:11%
        param = struct();
        param.reg_type = 'cgl';
        param.max_iter = 50;
        param.damped = 0;
        param.alpha = alpha_cgl(i);
        param.beta = beta_cgl(j);
        param.gamma = 0;
        param.rho = 100;
        % GL-SigRep
%         [~,Y,L_egilmez,offset,L_iter] = gl_poisson_log(X_noisy,param);
        [~,Y,L_egilmez,offset,L_iter] = gl_poisson_identity_interior(X_noisy,param);
        Ycell_poiss_cgl1{i,j} = Y;
        for k = 1:10
            if k <= size(L_iter,3)
                L = L_iter(:,:,k);
                Lcell_poiss_cgl1{i,j,k} = L;
                L(abs(L)<10^(-4))=0;
                if any(isnan(L))
                    continue
                end
                [precision_poiss_cgl1(i,j,k),recall_poiss_cgl1(i,j,k),Fmeasure_poiss_cgl1(i,j,k),NMI_poiss_cgl1(i,j,k),num_of_edges_poiss_cgl1(i,j,k)] = graph_learning_perf_eval(L_0,L);
            else
                precision_poiss_cgl1(i,j,k) = precision_poiss_cgl1(i,j,size(L_iter,3));
                recall_poiss_cgl1(i,j,k) = recall_poiss_cgl1(i,j,size(L_iter,3));
                Fmeasure_poiss_cgl1(i,j,k) = Fmeasure_poiss_cgl1(i,j,size(L_iter,3));
                NMI_poiss_cgl1(i,j,k) = NMI_poiss_cgl1(i,j,size(L_iter,3));
                num_of_edges_poiss_cgl1(i,j,k) = num_of_edges_poiss_cgl1(i,j,size(L_iter,3));
            end
        end
    end
end


%% performance
result1_poiss_l2(:,:,:,ii) = precision_poiss_l2;
result2_poiss_l2(:,:,:,ii) = recall_poiss_l2;
result3_poiss_l2(:,:,:,ii) = Fmeasure_poiss_l2;
result4_poiss_l2(:,:,:,ii) = NMI_poiss_l2;
result5_poiss_l2(:,:,:,ii) = num_of_edges_poiss_l2;

result1_harvard_poiss(:,:,:,ii) = precision_harvard_poiss;
result2_harvard_poiss(:,:,:,ii) = recall_harvard_poiss;
result3_harvard_poiss(:,:,:,ii) = Fmeasure_harvard_poiss;
result4_harvard_poiss(:,:,:,ii) = NMI_harvard_poiss;
result5_harvard_poiss(:,:,:,ii) = num_of_edges_harvard_poiss;
% 
% result1_poiss_log(:,:,:,ii) = precision_poiss_log;
% result2_poiss_log(:,:,:,ii) = recall_poiss_log;
% result3_poiss_log(:,:,:,ii) = Fmeasure_poiss_log;
% result4_poiss_log(:,:,:,ii) = NMI_poiss_log;
% result5_poiss_log(:,:,:,ii) = num_of_edges_poiss_log;
% 
% result1_kalofolias_poiss(:,:,:,ii) = precision_kalofolias_poiss;
% result2_kalofolias_poiss(:,:,:,ii) = recall_kalofolias_poiss;
% result3_kalofolias_poiss(:,:,:,ii) = Fmeasure_kalofolias_poiss;
% result4_kalofolias_poiss(:,:,:,ii) = NMI_kalofolias_poiss;
% result5_kalofolias_poiss(:,:,:,ii) = num_of_edges_kalofolias_poiss;
% 
% result1_poiss_cgl1(:,:,ii) = precision_poiss_cgl1;
% result2_poiss_cgl1(:,:,ii) = recall_poiss_cgl1;
% result3_poiss_cgl1(:,:,ii) = Fmeasure_poiss_cgl1;
% result4_poiss_cgl1(:,:,ii) = NMI_poiss_cgl1;
% result5_poiss_cgl1(:,:,ii) = num_of_edges_poiss_cgl1;
% 
% result1_egilmez_poiss1(:,:,ii) = precision_egilmez_poiss1;
% result2_egilmez_poiss1(:,:,ii) = recall_egilmez_poiss1;
% result3_egilmez_poiss1(:,:,ii) = Fmeasure_egilmez_poiss1;
% result4_egilmez_poiss1(:,:,ii) = NMI_egilmez_poiss1;
% result5_egilmez_poiss1(:,:,ii) = num_of_edges_egilmez_poiss1;

result1_poiss_cgl1(:,:,:,ii) = precision_poiss_cgl1;
result2_poiss_cgl1(:,:,:,ii) = recall_poiss_cgl1;
result3_poiss_cgl1(:,:,:,ii) = Fmeasure_poiss_cgl1;
result4_poiss_cgl1(:,:,:,ii) = NMI_poiss_cgl1;
result5_poiss_cgl1(:,:,:,ii) = num_of_edges_poiss_cgl1;

result1_egilmez_poiss1(:,:,:,ii) = precision_egilmez_poiss1;
result2_egilmez_poiss1(:,:,:,ii) = recall_egilmez_poiss1;
result3_egilmez_poiss1(:,:,:,ii) = Fmeasure_egilmez_poiss1;
result4_egilmez_poiss1(:,:,:,ii) = NMI_egilmez_poiss1;
result5_egilmez_poiss1(:,:,:,ii) = num_of_edges_egilmez_poiss1;


graph_poiss_l2{ii} = Lcell_poiss_l2;
% graph_harvard_poiss{ii} = Lcell_harvard_poiss;
% graph_poiss_log{ii} = Lcell_poiss_log;
% graph_kalofolias_poiss{ii} = Lcell_kalofolias_poiss;
graph_poiss_cgl1{ii} = Lcell_poiss_cgl1;
% graph_egilmez_poiss1{ii} = Lcell_egilmez_poiss1;

y_poiss_l2{ii} = Ycell_poiss_l2;
% offset_poiss_l2{ii} = Ocell_poiss_l2;
% y_poiss_log{ii} = Ycell_poiss_log;
% offset_poiss_log{ii} = Ocell_poiss_log;
y_poiss_cgl1{ii} = Ycell_poiss_cgl1;
% offset_poiss_cgl1{ii} = Ocell_poiss_cgl1;

end
% save("poisson_er02_t2000_exp20_p4p2.mat");