%% GLEN demo
clear;close all
addpath("../GLEN");

%% load
chicago_crime_data = load("tensor_data_chicago_crime-master/chicago_crime.mat");
load("crime_name_abbre.mat");

%% pre-processing
num_years = 10;
num_areas = length(chicago_crime_data.crime_mode_3_area);
num_crimes = length(chicago_crime_data.crime_mode_4_type);

crime_tensor = collapse(chicago_crime_data.crime_tensor(end-356*num_years+1:end,:,:,:),2);
X_noisy = tenmat(crime_tensor,3,[2,1]).data;
X_noisy = sum(reshape(X_noisy,[num_crimes,num_areas,356,num_years]),3);
X_noisy = reshape(X_noisy,[num_crimes,num_areas*num_years]);
X_noisy = X_noisy(sum(X_noisy,2)>0,:);

%% cgl

% beta_cgl = 10.^[-1:-0.3:-4]; % grid search
beta_cgl = 10^(-2.2);

S = cov(log(X_noisy+1)');
A_mask = ones(size(S))-eye(size(S));

for j = 1:length(beta_cgl)
    beta = beta_cgl(j);
    L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,50,1);
    Lcell_poiss_cgl1{j} = L;
    L(abs(L)<10^(-4))=0;
end
gplot_chicago_crimes(L,nodenames);

%% icassp

mu = mean(X_noisy,2);
sigma = cov(X_noisy');
S = ((sigma-diag(mu))./mu)./mu'+1;
S = log(S);
A_mask = ones(size(S))-eye(size(S));

beta_icassp = 0.79;
% beta_icassp = 10.^[0.1:-0.05:-0.85]; % grid search

for j = 1:length(beta_icassp)
    beta = beta_icassp(j);
    L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,50,1);
    Lcell_poiss_cgl1_icassp{j} = L;
    L(abs(L)<10^(-4))=0;
end
gplot_chicago_crimes(L,nodenames);

%% GLEN

alpha_glen = 5;
beta_glen = 0.05;
% alpha_glen = [10,5,1,0.5,0.1,0.05]; % grid search
L_init = estimate_cgl(cov(log(X_noisy+1)'),A_mask,10^(-2.5),1e-4,1e-6,50,1);

for i = 1:length(alpha_glen)
    for j = 1:length(beta_glen)
        param = struct();
        param.reg_type = 'cgl';
        param.max_iter = 20;
        param.alpha = alpha_glen(i);
        param.beta = beta_glen(j);
        param.gamma = 0;
        param.vi = 0;
        param.L_init = L_init;
        [L,Y,offset,L_iter,O_iter] = glen_poisson(X_noisy, param);
        for k = 1:20
            if k <= size(L_iter,3)
                L = L_iter(:,:,k);
            else
                L = L_iter(:,:,end);
            end
            Lcell_poiss_glen{i,j,k} = L;
        end
        Ycell_poiss_glen{i,j} = Y;
    end
end
gplot_chicago_crimes(L,nodenames);


