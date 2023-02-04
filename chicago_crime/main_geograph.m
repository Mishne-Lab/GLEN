%% load
chicago_crime_data = load("tensor_data_chicago_crime-master/chicago_crime.mat");
load("crime_name_abbre.mat");
% crime_tensor = collapse(chicago_crime_data.crime_tensor(end-355:end,:,:,:),[1,2]);
% X_noisy = tenmat(crime_tensor,2).data;
% X_noisy = X_noisy(sum(X_noisy,2)>0,:);

years = 10;
crime_tensor = collapse(chicago_crime_data.crime_tensor(end-356*years+1:end,:,:,:),2);
X_noisy = tenmat(crime_tensor,3,[2,1]).data;
X_noisy = sum(reshape(X_noisy,[32,77,356,years]),3);
X_noisy = reshape(X_noisy,[32,77*years]);
X_noisy = X_noisy(sum(X_noisy,2)>0,:);

% crime_tensor = collapse(chicago_crime_data.crime_tensor(500:end,:,:,[31,3,18,7]),2); % sum over dates
% crime_tensor = chicago_crime_data.crime_tensor(end-356:end,:,:,31); % sum over dates
% X_noisy = tenmat(crime_tensor,[1,3]).data;
% crime_tensor = collapse(chicago_crime_data.crime_tensor,[2,4]); % sum over hours & types
% crime_tensor = collapse(chicago_crime_data.crime_tensor(end-356:end,:,:,:),4); % sum over dates
% crime_tensor = collapse(chicago_crime_data.crime_tensor(end-356:end,:,:,:),1); % sum over dates


%% dong

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
% % result1_poiss_l2(:,ii) = precision_poiss_l2;
% % result2_poiss_l2(:,ii) = recall_poiss_l2;
% % result3_poiss_l2(:,ii) = Fmeasure_poiss_l2;
% % result4_poiss_l2(:,ii) = NMI_poiss_l2;
% % result5_poiss_l2(:,ii) = num_of_edges_poiss_l2;
% % result6_poiss_l2(:,ii) = rel_err_poiss_l2;
% % 
% % graph_poiss_l2{ii} = Lcell_poiss_l2;

%% kalofolias

% % weighted, normalized, offset = 2, -2
% beta_log = 10.^[0.5:-0.1:-0.5];
% gamma_log = 10.^[-0:-0.1:-1];
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
% % result1_poiss_log(:,:,ii) = precision_poiss_log;
% % result2_poiss_log(:,:,ii) = recall_poiss_log;
% % result3_poiss_log(:,:,ii) = Fmeasure_poiss_log;
% % result4_poiss_log(:,:,ii) = NMI_poiss_log;
% % result5_poiss_log(:,:,ii) = num_of_edges_poiss_log;
% % result6_poiss_log(:,:,ii) = rel_err_poiss_log;
% % 
% % graph_poiss_log{ii} = Lcell_poiss_log;

%% cgl

% er sbm offset = 2, -2
beta_cgl = 10.^[-1:-0.1:-2]; % log
beta_cgl = 10.^[0.2:-0.1:-0.2]; % last year, decent results
beta_cgl = 10.^[0:-0.1:-1]; % more signals

beta_cgl = 10.^[-1:-0.3:-4];

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

for j = 5%1:length_beta%10%5:15%5:11%
    beta = beta_cgl(j);
    % GL-SigRep
    L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,50,1);
    Lcell_poiss_cgl1{j} = L;
    L(abs(L)<10^(-4))=0;
%     [precision_poiss_cgl1(j),recall_poiss_cgl1(j),Fmeasure_poiss_cgl1(j),NMI_poiss_cgl1(j),num_of_edges_poiss_cgl1(j)] = graph_learning_perf_eval(L_0,L);
%     rel_err_poiss_cgl1(j) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
end

% result1_poiss_cgl1(:,ii) = precision_poiss_cgl1;
% result2_poiss_cgl1(:,ii) = recall_poiss_cgl1;
% result3_poiss_cgl1(:,ii) = Fmeasure_poiss_cgl1;
% result4_poiss_cgl1(:,ii) = NMI_poiss_cgl1;
% result5_poiss_cgl1(:,ii) = num_of_edges_poiss_cgl1;
% result6_poiss_cgl1(:,ii) = rel_err_poiss_cgl1;
% 
% graph_poiss_cgl1{ii} = Lcell_poiss_cgl1;

%% cgl-ours

% sbm offset = 2, -2
% beta_cgl = 10.^[-0.5:-0.1:-1.5]; % design 1
% beta_cgl = 10.^[0.5:-0.1:-0.5]; % design 1
beta_cgl = 10.^[0.5:-0.1:-0.5]; % last year, decent results
beta_cgl = 10.^[0.1:-0.05:-0.85];

length_beta = length(beta_cgl);

% X_noisy = X_noisy(:,end-76:end);
mu = mean(X_noisy,2);
sigma = cov(X_noisy');
S = ((sigma-diag(mu))./mu)./mu'+1;
% S(S<1e-2) = 1e-2;
S = log(S);
A_mask = ones(size(S))-eye(size(S));

for j = 5%1:length_beta%10%5:15%5:11%
    beta = beta_cgl(j);
    % GL-SigRep
%     L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,50,1); % 0.79
    L = estimate_cgl(S,A_mask,0.79,1e-4,1e-6,50,1);
    Lcell_poiss_cgl1_icassp{j} = L;
    L(abs(L)<10^(-4))=0;
%     [precision_poiss_cgl1_new(j),recall_poiss_cgl1_new(j),Fmeasure_poiss_cgl1_new(j),NMI_poiss_cgl1_new(j),num_of_edges_poiss_cgl1_new(j)] = graph_learning_perf_eval(L_0,L);
%     rel_err_poiss_cgl1_new(j) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
end

% result1_poiss_cgl1_new(:,ii) = precision_poiss_cgl1_new;
% result2_poiss_cgl1_new(:,ii) = recall_poiss_cgl1_new;
% result3_poiss_cgl1_new(:,ii) = Fmeasure_poiss_cgl1_new;
% result4_poiss_cgl1_new(:,ii) = NMI_poiss_cgl1_new;
% result5_poiss_cgl1_new(:,ii) = num_of_edges_poiss_cgl1_new;
% result6_poiss_cgl1_new(:,ii) = rel_err_poiss_cgl1_new;
% 
% graph_poiss_cgl1_new{ii} = Lcell_poiss_cgl1_new;

%% poisson alternating

beta_cgl = 10.^[0.5:-0.1:-0.5]; % last year, decent results
beta_cgl = 10.^[-0.5:-0.2:-2.5];
beta_cgl = 10.^[-1:-0.2:-3]; 
alpha_cgl = 10.^[1:-0.2:-1]; % 5 for vi0.1 okay reg1
alpha_cgl = 10.^[2:-0.2:0];

alpha_cgl = 10.^[2:-0.5:-2];
beta_cgl = 10.^[1:-0.5:-3]; 
%%
beta_cgl = 0.05;
alpha_cgl = [10,5,1,0.5,0.1,0.05];

length_beta = length(beta_cgl);
length_alpha = length(alpha_cgl);
L_init = estimate_cgl(cov(log(X_noisy+1)'),A_mask,10^(-2.5),1e-4,1e-6,50,1);
%%

for i = 2%1:length_alpha
    for j = 1:length_beta%10%5:15%5:11%
        % GL-SigRep
        param = struct();
        param.reg_type = 'cgl';
        param.max_iter = 20;
        param.alpha = alpha_cgl(i);
        param.beta = beta_cgl(j);%*1.5;
        param.gamma = 0;%2
        %         L_init = eye(31)-1/31;
        param.L_init = L_init;
        [L,Y,offset,L_iter,O_iter] = vgl_poisson_log(X_noisy, param);
        for k = 1:20
            if k <= size(L_iter,3)
                L = L_iter(:,:,k);
            else
                L = L_iter(:,:,end);
            end
            Lcell_poiss_cgl1_new{i,j,k} = L;
        end
        Ycell_poiss_cgl1_new{i,j} = Y;
    end
end