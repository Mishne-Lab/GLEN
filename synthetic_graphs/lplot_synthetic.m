function lplot_synthetic(graphs, Fmeasure_grid, rel_err_grid, eval, save)
%% search for best parameters
N = 20;
f = squeeze(mean(Fmeasure_grid,2));
re = squeeze(mean(rel_err_grid,2));
[max_val,max_idx_sp] = max(f,[],'all');
ahh = re(f>=(max_val-0.02));
max_idx_wp = find(re==min(ahh));

%% save the Laplacian plots
switch eval
    case 'sp' % structure prediction
        for ii = 1:20
            final_graphs = graphs{ii};
            L = final_graphs{max_idx_sp};
            L(abs(L)<10^(-4))=0;
            L = L - diag(sum(L,1));
            L = L/trace(L)*N;
            imagesc(L);
            c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;
            exportgraphics(ax,save,'Resolution',300)
        end
    case 'wp' % weight prediction
        for ii = 1:20
            final_graphs = graphs{ii};
            L = final_graphs{max_idx_wp};
            L(abs(L)<10^(-4))=0;
            L = L - diag(sum(L,1));
            L = L/trace(L)*N;
            imagesc(L);
            c=colorbarpwn(-1, 2);c.FontSize=20;c.Position = c.Position + 1e-10;set(gca,'XTick',[]);set(gca,'YTick',[]);axis square;axis square;ax=gca;
            exportgraphics(ax,save,'Resolution',300)
        end
end

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
