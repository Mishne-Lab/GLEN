%% typical mean firing analysis
mean_firing = squeeze(mean(spikes,2));
MdlLinear = fitcdiscr(mean_firing,tar);
res = predict(MdlLinear,mean_firing);
acc_mf = sum(res==tar')/length(mean_firing);

%% laplacian
laplacian = Acell;
laplacian = laplacian(:,sum(laplacian,1)>0);
MdlLinear = fitcdiscr(laplacian,tar);
res = predict(MdlLinear,laplacian);
acc_laplacian = sum(res==tar')/length(laplacian);

%% degree
deg = Dcell;
deg = deg(:,sum(deg,1)>0);
MdlLinear = fitcdiscr(deg,tar');
res = predict(MdlLinear,deg);
acc_deg = sum(res==tar')/length(deg);

%% plot denoised firing rate
Ocell(Ocell==0) = -inf;
for i = 0:7
    denoised = exp(Ycell+reshape(Ocell,[length(Ocell),1,size(Ocell,2)]));
    mask = (bump==1)&(tar==i*45);
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