%% Load the graph and signals
clear;close all
load('area2_bump_train_spikes_resampled.mat');
load('area2_bump_trial_info.mat');
%%
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

alpha = 10.^[-1:-0.5:-3];
% beta = 10.^[0:-0.1:-2];
beta = 10.^[-1.5:-0.5:-3.5];
gamma = 10.^[-1:-0.5:-3];
% 
% alpha = 10^(-1.5);
% beta = 10^(-2);
% gamma = 10^(-1);

alpha = 10.^[-1.1:-0.2:-1.9];
beta = 10.^[-1.6:-0.2:-2.4];
gamma = 10.^[-1:-0.5:-3];

% alpha = 10.^[-0.6:-0.2:-1.4];
% beta = 10.^[-1:-0.2:-2];

% alpha = 0.04;
% beta = 0.01;
% gamma = 0.01;
gamma = 0;

%% main loop

for i = 2%1:length(alpha)%2%
    for j = 1%1:length(beta)%1%
        for k = 1%1:length(gamma)
            param.reg_type = 'cgl';
            param.alpha = alpha(i);
            param.beta = beta(j);
            param.gamma = gamma(k);%*2;
%             param.trace = trace(k);


            [L,Y,offset,L_iter,O_iter] = gl_tv_poisson_log(X_noisy, param);
            if any(isnan(L), 'all')
                convergence(t) = 0;
                continue
            end
%             A = -L+diag(diag(L));
%             imagesc(A);
%             num_of_edge = sum(A>1E-4,'all')/2;
%             title(num2str(num_of_edge));

%             Lcell{i,j} = L;
%             Ocell{i,j} = offset;
%             Ycell{i,j} = Y;
            

            A = -L+diag(diag(L));
            A_full = zeros(65);
            A_full(active_nodes,active_nodes) = A;
            Acell(t,:) = squareform(A_full);

            Ycell(t,:,active_nodes) = Y';
            Dcell(t,active_nodes) = diag(L);

            fuck(t,active_nodes) = sum(L,1);
            Ocell(t,active_nodes) = offset;
        end
    end
end
end


%% clustering
% clusters = kmeans(Acell,8);

% [coeff,score,latent] = pca(Acell(bump==0,:));

figure(2)
% mean_firing = squeeze(mean(spikes,2));
% Ocell(Ocell==0) = -inf;
% mean_firing = squeeze(mean(exp(Ycell+reshape(Ocell,[length(Ocell),1,size(Ocell,2)])),2));
mean_firing = exp(Ocell);
[coeff,score,latent] = pca(mean_firing(bump==0,:));

% for t = 1:length(spikes)
%     c = cov(squeeze(spikes(t,:,:)));
%     c = c - diag(diag(c));
%     Ccell(t,:) = squareform(c);
% end
% [coeff,score,latent] = pca(Ccell(bump==0,:));
% 
% gscatter(score(:,1),score(:,2),tar(bump==0),parula(8));

scatter3(score(:,1),score(:,2),score(:,3),10,tar(bump==0),'filled')
% 
% deg = zeros(length(spikes),65);
% for t = 1:length(spikes)
%     deg(t,:) = sum(squareform(Acell(t,:)),1);
% end
% [coeff,score,latent] = pca(deg(bump==0,:));
% gscatter(score(:,1),score(:,2),tar(bump==0),parula(8));

% Lcell_tv_clustering = zeros(length(Lcell_tv_trials),size(spikes,3)^2);
% for i = 1:length(Lcell_tv_trials)
%     Lcell_tv_clustering(i,:) = Lcell_tv_trials{i}(:);
% end
% clusters = kmeans(Lcell_tv_clustering(bump==0,:),8);
% nmi = perfeval_clus_nmi(double(clusters),double(tar(bump==0)));
% [coeff,score,latent] = pca(Lcell_tv_clustering(bump==0,:));
% % scatter(score(:,1),score(:,2),[],tar(bump==0),'filled');
% gscatter(score(:,1),score(:,2),tar(bump==0),parula(8));
% set(gca,'XTick',[]);
% set(gca,'YTick',[]);
% print(gcf,'graph_clustering.png','-dpng','-r300');
% scatter3(score1(:,1),score1(:,2),score1(:,3),[],tar(bump==0));
% Lcell_clustering = zeros(length(Lcell_trials),size(spikes_smth,3)^2);
% for i = 1:length(Lcell_trials)
%     Lcell_clustering(i,:) = Lcell_trials{i}(:);
% end
% [coeff,score,latent] = pca(Lcell_clustering(bump==0,:));
% [coeff,score,latent] = pca(spike_sum(bump==0,:));
% gscatter(score(:,1),score(:,2),tar(bump==0),parula(8));