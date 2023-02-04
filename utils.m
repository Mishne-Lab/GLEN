%% plot trend with more iterations
% for i = 1:length(alpha_cgl)%length(alpha_l2)%
%     for j = 1:length(beta_cgl)%length(beta_l2)%
%         subplot(length(alpha_cgl),length(beta_cgl),(i-1)*length(beta_cgl)+j)
%         plot(squeeze(mean(result3_poiss_cgl1(i,j,:,:),4)));
% %         plot(squeeze(Fmeasure_poiss_cgl1(i,j,:)));
%         ylim([0.5,0.75]);
% %         ylim([0.3,0.5]);
%         xlim([0,10]);
%     end
% end

for i = 1:length(alpha_cgl)%length(alpha_l2)%
    for j = 1:length(beta_cgl)%length(beta_l2)%
        subplot(length(alpha_cgl),length(beta_cgl),(i-1)*length(beta_cgl)+j)
%         plot(squeeze(mean(result3_poiss_l2(i,j,:,:),4)));
        plot(squeeze(mean(result3_poiss_cgl1(i,j,:,:),4)));
        yline(0.8262);
        ylim([0,1]);
        xlim([0,20])
    end
end
%%
plot(mean(result3_poiss_cgl1(1,:,1,:),4));
ylabel('f-score');
xlabel('ln(beta)')
xlim([1,21]);
xticklabels([-0:-0.2:-2]);
yline(0.8262);
%%
% imagesc(max(mean(result3_poiss_l2,4),[],3));
imagesc(squeeze(mean(result3_poiss_cgl1(:,:,end,:),4)));
% imagesc(squeeze(mean(result3_poiss_dong(:,:,1,:),4)));

%% 
for i = 1:length(alpha_cgl)
    for j = 1:length(beta_cgl)
        subplot(length(alpha_cgl),length(beta_cgl),(j-1)*length(beta_cgl)+i);
        laplacian = graph_poiss_cgl1{2}{i,j,4};
        graph = diag(diag(laplacian)) - laplacian;
        imagesc(graph);
    end
end

%%
for k = 1:10
    subplot(2,5,k);
    imagesc(mean(result3_poiss_cgl1(:,:,k,:),4));
    colorbar();
end

%%
nreplicate = 20;
for ii = 1:nreplicate

X = data{ii,3};
% X = X*0.2+1;
% X(X<-5) = -5+1e-4;
offset = [2*ones(10,1);-4*ones(10,1)];

X_noisy = poissrnd(exp(X+offset), size(X));

% % data{ii,1} = A;
% % data{ii,2} = L_0;
% data{ii,3} = data{ii,3};
data{ii,4} = X_noisy;

end

%%
squeeze(mean(result3_poiss_cgl1(:,:,end,:),4));

%%
f = squeeze(mean(result3_poiss_cgl1(:,:,end,:),4));
re = squeeze(mean(result6_poiss_cgl1(:,:,end,:),4));
% f = squeeze(mean(result3_poiss_log,3));
% re = squeeze(mean(result6_poiss_log,3));
max_val = max(f,[],'all')
ahh = re(f>=(max_val-0.02));
min(ahh)

%%
for k = 1:20
    subplot(4,5,k)
    imagesc(L_iter(:,:,2*k));
end

%%
for k = 1:20
%     norm(L_iter(:,:,k)-L_0, 'fro') / norm(L_0, 'fro')
    subplot(4,5,k)
    imagesc(data{k,2});
end

%%
% fuck = zeros(20,20);
figure(2)
wat = 0;
for ii = 1:20
%     L = graph_poiss_cgl1{ii}{10,7,end};
    L = graph_poiss_log{ii}{10,10};
%     L = L - diag(sum(L,1));
%     L = L/trace(L)*N;
%     L = graph_poiss_l2{ii}{3};
    L_0 = data{ii,2};
    wat = wat + norm(L-L_0, 'fro') / norm(L_0, 'fro') / 20;
    fuck(:,ii) = sum(L,2);
    subplot(4,5,ii)
    imagesc(L);
end
wat

%%
for ii = 1:20
    L_0 = data{ii,2};
    for i = 1:11
        for j = 1:11
            L = graph_poiss_cgl1{ii}{i,j,end};
            L = L/trace(L)*N;
            deg = diag(L);
            deg_0 = diag(L_0);
            edge = squareform(-L+diag(deg));
            edge_0 = squareform(-L_0+diag(deg_0));
            rel_err_edge1(i,j,ii) = norm(edge-edge_0, 1) / norm(edge_0, 1);
            rel_err_edge2(i,j,ii) = norm(edge-edge_0, 2) / norm(edge_0, 2);
            rel_err_deg1(i,j,ii) = norm(deg-deg_0, 1) / norm(deg_0, 1);
            rel_err_deg2(i,j,ii) = norm(deg-deg_0, 2) / norm(deg_0, 2);
            rel_err(i,j,ii) = norm(L-L_0, 'fro') / norm(L_0, 'fro');
%     fuck(:,ii) = sum(L,2);
%     subplot(4,5,ii)
%     imagesc(L-L_0);
        end
    end
end
%%
squeeze(mean(rel_err,3));


%% ablation studies
Ts = [500,1000,2000];
methods = ["l2","log","cgl1","cgl1_new"];
fs = zeros(length(Ts),length(methods));
nmi = zeros(length(Ts),length(methods));
relerr = zeros(length(Ts),length(methods));
for i = 1:length(Ts)
    for j = 1:length(methods)
        t = Ts(i);
        m = methods(j);
        load(['weighted_normalized_results_er03_n20_t', num2str(t), '_offset2-2'],['result3_poiss_',char(m)],['result4_poiss_',char(m)],['result6_poiss_',char(m)]);
        if j == 2
            d = 3;
        else
            d = 2;
        end
        [val,idx] = max(mean(eval(['result3_poiss_',char(m)]),d),[],'all');
        fs(i,j) = val;
        vals = mean(eval(['result4_poiss_',char(m)]),d);
        nmi(i,j) = vals(idx);
        vals = mean(eval(['result6_poiss_',char(m)]),d);
        relerr(i,j) = vals(idx);
    end
end


%%
[f2_val, idx] = max(mean(result3_poiss_log,3),[],'all')
% nmi_val = mean(result4_poiss_l2(idx,:),2)

%%
% imagesc(data{10,2});c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;exportgraphics(ax,'./figures/weighted_normalized_gt_er03_n20_t2000_offset2-2.jpg','Resolution',300)
imagesc(graph_poiss_cgl1{10}{1});c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;exportgraphics(ax,'./figures/weighted_normalized_cgl_5it_er03_n20_t2000_offset2-2.jpg','Resolution',300)
imagesc(graph_poiss_cgl1_new{10}{6});c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;exportgraphics(ax,'./figures/weighted_normalized_ourscgl_5it_er03_n20_t2000_offset2-2.jpg','Resolution',300)
% imagesc(graph_poiss_l2{10}{6});c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;exportgraphics(ax,'./figures/weighted_normalized_l2_er03_n20_t2000_offset2-2.jpg','Resolution',300)
% imagesc(graph_poiss_log{10}{7,6});c=colorbarpwn(-1, 5);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;ax=gca;exportgraphics(ax,'./figures/weighted_normalized_log_er03_n20_t2000_offset2-2.jpg','Resolution',300)
% imagesc(graph_poiss_logdet{10}{7});c=colorbarpwn(-1, 9);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;exportgraphics(ax,'./figures/weighted_normalized_scgl_er03_n20_t2000_offset2-2.jpg','Resolution',300)

%%
N = 20;
% f = squeeze(mean(result3_poiss_cgl1(:,:,end,:),4));
% re = squeeze(mean(result6_poiss_cgl1(:,:,end,:),4));
% f = squeeze(mean(result3_poiss_log,3));
% re = squeeze(mean(result6_poiss_log,3));
f = squeeze(mean(result3_poiss_cgl1,2));
re = squeeze(mean(result6_poiss_cgl1,2));
[max_val,max_idx] = max(f,[],'all')
ahh = re(f>=(max_val-0.02));
max_idx2 = find(re==min(ahh));
for k = 1:20
%     subplot(4,5,k);
    final_graphs = graph_poiss_cgl1{k};%(:,:,end);
    L = final_graphs{max_idx};
    L(abs(L)<10^(-4))=0;
    L = L - diag(sum(L,1));
    L = L/trace(L)*N;
    imagesc(L);
    c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;
    exportgraphics(ax,strcat('./figures/ws/cgl/sp_cgl_3it_ws201_n20_t2000_offset2-2-graph',num2str(k),'.jpg'),'Resolution',300)
end
for k = 1:20
%     subplot(4,5,k);
    final_graphs = graph_poiss_cgl1{k};%(:,:,end);
    L = final_graphs{max_idx2};
    L(abs(L)<10^(-4))=0;
    L = L - diag(sum(L,1));
    L = L/trace(L)*N;
    imagesc(L);
    c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;
    exportgraphics(ax,strcat('./figures/ws/cgl/wp_cgl_3it_ws201_n20_t2000_offset2-2-graph',num2str(k),'.jpg'),'Resolution',300)
end


% f = squeeze(mean(result3_poiss_logdet,2));
% re = squeeze(mean(result6_poiss_logdet,2));
% [max_val,max_idx] = max(f,[],'all')
% ahh = re(f>=(max_val-0.02));
% max_idx2 = find(re==min(ahh));
% for k = 1:20
% %     subplot(4,5,k);
%     final_graphs = graph_poiss_logdet{k};%(:,:,end);
%     L = final_graphs{max_idx};
%     L(abs(L)<10^(-4))=0;
%     L = L - diag(sum(L,1));
%     L = L/trace(L)*N;
%     imagesc(L);
%     c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;
%     exportgraphics(ax,strcat('./figures/er/lake/sp_scgl_3it_er03_n20_t2000_offset2-2-graph',num2str(k),'.jpg'),'Resolution',300)
% end
% for k = 1:20
% %     subplot(4,5,k);
%     final_graphs = graph_poiss_logdet{k};%(:,:,end);
%     L = final_graphs{max_idx2};
%     L(abs(L)<10^(-4))=0;
%     L = L - diag(sum(L,1));
%     L = L/trace(L)*N;
%     imagesc(L);
%     c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;
%     exportgraphics(ax,strcat('./figures/er/lake/wp_scgl_3it_er03_n20_t2000_offset2-2-graph',num2str(k),'.jpg'),'Resolution',300)
% end
% for k = 1:20
%     L = data{k,2};
%     imagesc(L);
%     c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;
%     exportgraphics(ax,strcat('./figures/ws/gt/ws201_n20_t2000_offset2-2-graph',num2str(k),'.jpg'),'Resolution',300)
% end
