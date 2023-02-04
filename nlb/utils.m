%%
% Y = Ycell{4,2};
% L = Lcell{4,2};
A = -L+diag(diag(L));
idx = optimalleaforder(linkage(A,'single'),A);
imagesc(Y(idx,:));
%% viz
for i = 1:length(alpha)
    for j = 1:length(beta)
        subplot(length(alpha), length(beta), length(beta)*(i-1)+j);
        L = Lcell{i,j};
        A = -L+diag(diag(L));
        imagesc(A);
%         offset = Ocell{i,j};
%         offset(offset==0) = -inf;
%         imagesc(exp(Ycell{i,j}+offset));
%         num_of_edge = sum(A>1E-4,'all')/2;
%         title(num2str(num_of_edge));
    end
end
%% viz all
% tar0 = tar(bump==0);
% Acell0 = Acell(bump==0,:);
% [~,idx]=sort(tar0);
% imagesc(Acell0(idx,:));
% [~,edge_idx]=sort(mean(Acell0,1),'descend');
% imagesc(Acell0(idx,edge_idx));

% [~,idx]=sort(0.1*double(bump)+tar);
% ys = find(tar(idx(1:end-1))~=tar(idx(2:end)));
% imagesc(Acell(idx,:));
% yline(ys,'r','LineWidth',1);
% % [~,edge_idx]=sort(mean(Acell,1),'descend');
% % imagesc(Acell(idx,edge_idx));
% xlabel('edge');
% ylabel('trial');
% colormap gray
% 
% PD = pdist(Acell','hamming');
% edge_idx = optimalleaforder(linkage(Acell','single'),PD);
% imagesc(Acell(idx,edge_idx));

%% typical mean firing analysis
mean_firing = squeeze(mean(spikes,2));
MdlLinear = fitcdiscr(mean_firing,tar);
res = predict(MdlLinear,mean_firing);
acc = sum(res==tar')/length(mean_firing)

%% covariance
std_firing = squeeze(std(spikes,0,2));
MdlLinear = fitcdiscr(std_firing,tar);
res = predict(MdlLinear,std_firing);
acc = sum(res==tar')/length(std_firing)

%% offset
Ocell(Ocell==0) = -inf;
offset_firing = squeeze(mean(exp(Ycell+reshape(Ocell,[length(Ocell),1,size(Ocell,2)])),2));
MdlLinear = fitcdiscr(offset_firing,tar);
res = predict(MdlLinear,offset_firing);
acc = sum(res==tar')/length(offset_firing)

%%
Ocell(Ocell==0) = -inf;
offset_firing = exp(Ocell);
MdlLinear = fitcdiscr(offset_firing,tar);
res = predict(MdlLinear,offset_firing);
acc = sum(res==tar')/length(offset_firing)

%% laplacian
laplacian = Acell;
laplacian = laplacian(:,sum(laplacian,1)>0);
MdlLinear = fitcdiscr(laplacian,tar);
res = predict(MdlLinear,laplacian);
acc = sum(res==tar')/length(laplacian)

%% degree
deg = Dcell;
% deg = double(Dcell>0);
% deg = deg ./ sum(deg,2);
deg = deg(:,sum(deg,1)>0);
MdlLinear = fitcdiscr(deg,tar');
res = predict(MdlLinear,deg);
acc = sum(res==tar')/length(deg)

%%
deg = Dcell;
mean_firing = squeeze(mean(spikes,2));
% X = [deg,mean_firing];
X = deg;
% deg = double(Dcell>0);
% deg = deg ./ sum(deg,2);
X = X(:,sum(X,1)>0);
MdlLinear = fitcdiscr(X,tar');
res = predict(MdlLinear,X);
acc = sum(res==tar')/length(X)

%%
N = size(spikes,1);
idx = randperm(N);
section = round(N*0.05);
acc = 0;
deg = Dcell;
mean_firing = squeeze(mean(spikes,2));
std_firing = squeeze(std(spikes,0,2));
% mean_firing = squeeze(mean(spikes,2));
% X = [deg,mean_firing];
X = deg;
for i = 1:20
    train_idx = [idx(1:(i-1)*section),idx(i*section+1:end)];
    test_idx = idx((i-1)*section+1:i*section);
    X = X(:,sum(X,1)>0);
    MdlLinear = fitcdiscr(X(train_idx,:),tar(train_idx)');
    res = predict(MdlLinear,X(test_idx,:));
    acc = acc + sum(res==tar(test_idx)')/length(test_idx)/20;
end
acc

%%
N = size(spikes,1);
idx = randperm(N);
section = round(N*0.1);
acc = 0;
mean_firing = squeeze(mean(spikes,2));
std_firing = squeeze(std(spikes,0,2));
X = [mean_firing,std_firing];
for i = 1:10
    train_idx = [idx(1:(i-1)*section),idx(i*section+1:end)];
    test_idx = idx((i-1)*section+1:i*section);
    X = X(:,sum(X,1)>0);
    MdlLinear = fitcdiscr(X(train_idx,:),tar(train_idx)');
    res = predict(MdlLinear,X(test_idx,:));
    acc = acc + sum(res==tar(test_idx)')/length(test_idx)/10;
end
acc

%%
mean_firing = squeeze(mean(spikes,2));
std_firing = squeeze(std(spikes,0,2));
X = [mean_firing,std_firing];
% deg = double(Dcell>0);
% deg = deg ./ sum(deg,2);
X = X(:,sum(X,1)>0);
MdlLinear = fitcdiscr(X(train_idx,:),tar(train_idx)');
res = predict(MdlLinear,X(test_idx,:));
acc = sum(res==tar(test_idx)')/length(test_idx)

%% laplacian to degree
deg = zeros(size(Acell,1),size(spikes,3));
for i = 1:size(Acell, 1)
    deg(i,:) = sum(squareform(Acell(i,:)),1);
end
deg = deg(:,sum(deg,1)>0);
MdlLinear = fitcdiscr(deg,tar);
res = predict(MdlLinear,deg);
acc = sum(res==tar')/length(deg)

%%
mean_spike_nobump = squeeze(mean(spikes(bump==0,:,:),1))';
PD = pdist(mean_spike_nobump);
idx = optimalleaforder(linkage(PD,'single'),PD);
imagesc(mean_spike_nobump(idx,:));

%%
for i = 0:7
mean_spike_bump = squeeze(mean(spikes(bump==1,:,:),1))';
% PD = pdist(mean_spike_bump);
% idx = optimalleaforder(linkage(PD,'single'),PD);
imagesc(mean_spike_bump(idx,:));
colormap bone
grid off
xlabel('time/s');ylabel('neuron')
yticks([]);ax=gca;
ax.XAxis.FontSize = 24;
ax.YAxis.FontSize = 24;
exportgraphics(ax,strcat('./figures/mean_spike_tar',num2str(i*45),'_bump3.jpg'),'Resolution',300)
end

%%
rates = Ycell;
rates(Ycell==0) = -0.5;
% imagesc(squeeze(mean(rates(bump==0,:,:),1))');
mean_rate_bump = squeeze(mean(rates(bump==1,:,:),1))';
PD = pdist(mean_rate_bump);
idx = optimalleaforder(linkage(PD,'single'),PD);
imagesc(mean_rate_bump(idx,:));

mean_spike_0 = squeeze(mean(spikes(mask,:,:),1))';
PD = pdist(mean_spike_0);
% idx = optimalleaforder(linkage(PD,'single'),PD);
imagesc(mean_spike_0(idx,:));
colormap bone
%%
figure(2)
for i = 0:7
    Ocell(Ocell==0) = -inf;
    denoised = exp(Ycell+reshape(Ocell,[length(Ocell),1,size(Ocell,2)]));
    mask = (bump==1)&(tar==i*45);
    % imagesc(squeeze(mean(denoised(bump==0,:,:),1))');
    % imagesc(squeeze(mean(Ycell(bump==1,:,:),1))');
    mean_denoise_bump = squeeze(mean(denoised(mask,:,:),1))';
    PD = pdist(mean_denoise_bump);
    idx = optimalleaforder(linkage(PD,'single'),PD);
    imagesc(mean_denoise_bump(idx,:));
    colormap bone
    grid off
    yticks([]);
    xlabel('time/s', 'FontSize', 20);ylabel('neuron', 'FontSize', 20)
    ax=gca;
    ax.XAxis.FontSize = 24;
    ax.YAxis.FontSize = 24;
    exportgraphics(ax,strcat('./figures/nlb_tar',num2str(i*45),'_denoised_ourstv_reg1_a004_b001_g01.jpg'),'Resolution',300)

    mean_spike_bump = squeeze(mean(spikes(mask,:,:),1))';
    imagesc(mean_spike_bump(idx,:));
    colormap bone
    grid off
    xlabel('time/s');ylabel('neuron')
    yticks([]);ax=gca;
    ax.XAxis.FontSize = 24;
    ax.YAxis.FontSize = 24;
    exportgraphics(ax,strcat('./figures/mean_spike_tar',num2str(i*45),'_bump3.jpg'),'Resolution',300)

end

%% lfads
% Identify the datasets you'll be using
% Here we'll add one at ~/lorenz_example/datasets/dataset001.mat
dc = LorenzExperiment.DatasetCollection('~/lorenz_example/datasets');
dc.name = 'lorenz_example';
ds = LorenzExperiment.Dataset(dc, 'dataset001.mat'); % adds this dataset to the collection
dc.loadInfo; % loads dataset metadata

%%
% Run a single model for each dataset, and one stitched run with all datasets
runRoot = '~/lorenz_example/runs';
rc = LorenzExperiment.RunCollection(runRoot, 'example', dc);

% run files will live at ~/lorenz_example/runs/example/

% Setup hyperparameters, 4 sets with number of factors swept through 2,4,6,8
par = LorenzExperiment.RunParams;
par.spikeBinMs = 2; % rebin the data at 2 ms
par.c_co_dim = 0; % no controller outputs --> no inputs to generator
par.c_batch_size = 150; % must be < 1/5 of the min trial count
par.c_gen_dim = 64; % number of units in generator RNN
par.c_ic_enc_dim = 64; % number of units in encoder RNN
par.c_learning_rate_stop = 1e-3; % we can stop really early for the demo
parSet = par.generateSweep('c_factors_dim', [2 4 6 8]);
rc.addParams(parSet);

% Setup which datasets are included in each run, here just the one
runName = dc.datasets(1).getSingleRunName(); % == 'single_dataset001'
rc.addRunSpec(LorenzExperiment.RunSpec(runName, dc, 1));

% Generate files needed for LFADS input on disk
rc.prepareForLFADS();

% Write a python script that will train all of the LFADS runs using a
% load-balancer against the available CPUs and GPUs
rc.writeShellScriptRunQueue('display', 0, 'virtualenv', 'tensorflow');