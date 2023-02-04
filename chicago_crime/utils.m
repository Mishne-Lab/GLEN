%% viz laplacians
for i = 1:11%length(Lcell_poiss_cgl1_new)
    subplot(3,4,i);
    L = Lcell_poiss_cgl1{1,i};
    L = L/trace(L)*length(L);
    A = -L+diag(diag(L));
    imagesc(A);
    title(num2str(sum(L<-1E-4,'all')/2));
end
%%
% nodenames = chicago_crime_data.crime_mode_4_type;
nodenames = nodenames(sum(X_noisy,2)>0);
for i = 1:length(Lcell_poiss_cgl1)
    subplot(3,4,i);
    L = Lcell_poiss_cgl1{i};
    A = -L+diag(diag(L));
    g = graph(A);
%     g = graph(A*10, nodenames);
    plot(g,'LineWidth',g.Edges.Weight/10,'Layout','circle');
    title(num2str(num_of_edges(i)));
end

%%
% L = Lcell_poiss_cgl1_new{5};
% L = Lcell_poiss_cgl1_new{2,1,end};
% L = L_init;
% L = L_iter(:,:,1);
sum(L<-1E-4,'all')/2
L = L / trace(L) * length(L);
A = -L+diag(diag(L));
A(A<1E-4) = 0;
% idx = optimalleaforder(linkage(A,'single'),A);
idx = symrcm(A);
A = A(idx,idx);
g = graph(A);
% plot(g,'LineWidth',g.Edges.Weight*2,'Layout','circle','NodeFontSize',6);
plot(g,'LineWidth',g.Edges.Weight*5,'Layout','circle','NodeLabel',nodenames(idx),'NodeFontSize',8);
axis square;
axis off;
ax=gca;
% exportgraphics(ax,'./results/ours_cglinit_crime_graph_alpha5_beta005.jpg','Resolution',300)

%%
L = L_init;
L = L / trace(L) * length(L);
A = -L+diag(diag(L));
A(A<1E-4) = 0;
% idx = optimalleaforder(linkage(A,'single'),A);
% idx = symrcm(A);
A = A(idx,idx);
g = graph(A);
% plot(g,'LineWidth',g.Edges.Weight*2,'Layout','circle','NodeFontSize',6);
plot(g,'LineWidth',g.Edges.Weight*5,'Layout','circle','NodeLabel',nodenames(idx),'NodeFontSize',8);
axis square;
axis off;
ax=gca;
exportgraphics(ax,'./results/cgl_crime_graph_layout_alpha-25.jpg','Resolution',300)

%%
for i = 1:length_alpha
    for j = 1:length_beta
        subplot(length_alpha,length_beta,(i-1)*length_beta+j);
        L = Lcell_poiss_cgl1_new{i,j,20};
        imagesc(-L+diag(diag(L)));
        title(num2str(sum(L<-1E-4,'all')/2));
    end
end

%% viz graphs
% nodenames = chicago_crime_data.crime_mode_4_type;
nodenames = nodenames(sum(X_noisy,2)>0);
for i = 1:length(Lcell_poiss_cgl1_new)
    subplot(4,5,i);
    L = Lcell_poiss_cgl1_new{i};
    A = -L+diag(diag(L));
    idx = optimalleaforder(linkage(A),A);
    A = A(idx,idx);
    g = graph(A);
%     g = graph(A*10, nodenames);
%     plot(g,'LineWidth',g.Edges.Weight);
    plot(g,'LineWidth',exp(g.Edges.Weight),'Layout','circle','NodeLabel',nodenames(idx),'NodeFontSize',4);
%     title(num2str(num_of_edges(i)));
end

%%
nodenames = nodenames(sum(X_noisy,2)>0);
for i = 6:length_alpha-1%1:
    for j = 3:length_beta
        subplot(5,length_beta-2,(i-1-5)*(length_beta-2)+j-2);
        L = Lcell_poiss_cgl1_new{i,j};
        A = -L+diag(diag(L));
        idx = optimalleaforder(linkage(A),A);
        A = A(idx,idx);
        g = graph(A);
    %     g = graph(A*10, nodenames);
    %     plot(g,'LineWidth',g.Edges.Weight);
        plot(g,'LineWidth',g.Edges.Weight/10,'Layout','circle','NodeLabel',nodenames(idx),'NodeFontSize',4);
    %     title(num2str(num_of_edges(i)));
    end
end
%%
% L = Lcell_poiss_cgl1_new{5,4};
L = Lcell_poiss_cgl1{1,5};
L = L / trace(L) * 31;
A = -L+diag(diag(L));
idx = optimalleaforder(linkage(A),A);
A = A(idx,idx);
g = graph(A);
plot(g,'LineWidth',g.Edges.Weight,'Layout','circle','NodeLabel',nodenames(idx),'NodeFontSize',8);
% alpha_cgl(5)
% beta_cgl(4)

%%
figure(2)
for i = 1:9
    for j = 1:9
        subplot(9,9,9*(i-1)+j);
        L = Lcell_poiss_cgl1_new{i,j,end};
        imagesc(L);
        title(num2str(sum(L<-1E-4,'all')/2));
%         A = -L+diag(diag(L));
%         A = A(idx,idx);
%         g = graph(A);
%         plot(g,'LineWidth',g.Edges.Weight/10,'Layout','circle','NodeLabel',nodenames(idx),'NodeFontSize',4);
    end
end

%%
Y = Ycell_poiss_cgl1_new{4,6};
imagesc((Y*Y')/770);

%%
L = Lcell_poiss_cgl1_new{4,4};
A = -L+diag(diag(L));
idx = optimalleaforder(linkage(A),A);
A = A(idx,idx);
g = graph(A);
plot(g,'LineWidth',g.Edges.Weight,'Layout','circle','NodeLabel',nodenames(idx),'NodeFontSize',8);
axis square;
axis off;
ax=gca;

%% ours save fig
L = Lcell_poiss_cgl1_icassp{5};
A = -L+diag(diag(L));
idx = optimalleaforder(linkage(A),A);
A = A(idx,idx);
g = graph(A);
plot(g,'LineWidth',g.Edges.Weight*10,'Layout','circle','NodeLabel',nodenames(idx),'NodeFontSize',8);
axis square;
axis off;
ax=gca;
% exportgraphics(ax,'./results/ourscgl_crime_graph_5it_alpha-01.jpg','Resolution',300)

%% cgl save fig
% L = Lcell_poiss_cgl1{10}; % icassp first submission
L = Lcell_poiss_cgl1{5};
A = -L+diag(diag(L));
% idx = optimalleaforder(linkage(A),A);
A = A(idx,idx);
g = graph(A);
plot(g,'LineWidth',g.Edges.Weight/10,'Layout','circle','NodeLabel',nodenames(idx),'NodeFontSize',8);
axis square;
axis off;
ax=gca;
exportgraphics(ax,'./results/cgl_crime_graph_5it_alpha-25.jpg','Resolution',300)

%% count num of edges
for i = 1:length(Lcell_poiss_cgl1)
    L = Lcell_poiss_cgl1{i}<0;
    num_of_edges(i) = sum(L(:))/2;
end

%% hist noisy signals
for j = 1:12%size(X_noisy,1)
    subplot(3,4,j)
    hist(X_noisy(j,:),64);
end

%%

% get inferred graph
% L = Lcell_poiss_cgl1{2};
L = Lcell_poiss_cgl1_new{11};
G = -L+diag(diag(L));
G(G<0.5) = 0;
[row,col]=find(triu(G));

% load geographic data
M = shaperead('tensor_data_chicago_crime-master/geodata/chicago_geo_data.shp', 'UseGeoCoords', true);
axesm mercator
geoshow(M);

% overlay the graph
for i = 1:length(row)
    loc1 = mean(M(row(i)).BoundingBox);
    loc2 = mean(M(col(i)).BoundingBox);
    geoshow([loc1(2),loc2(2)],[loc1(1),loc2(1)],'LineWidth',G(row(i),col(i)));
end

%%
h = WattsStrogatz(20,2,0.1);
plot(h,'Layout','circle')
function h = WattsStrogatz(N,K,beta)
% H = WattsStrogatz(N,K,beta) returns a Watts-Strogatz model graph with N
% nodes, N*K edges, mean node degree 2*K, and rewiring probability beta.
%
% beta = 0 is a ring lattice, and beta = 1 is a random graph.

% Connect each node to its K next and previous neighbors. This constructs
% indices for a ring lattice.
s = repelem((1:N)',1,K);
t = s + repmat(1:K,N,1);
t = mod(t-1,N)+1;

% Rewire the target node of each edge with probability beta
for source=1:N    
    switchEdge = rand(K, 1) < beta;
    
    newTargets = rand(N, 1);
    newTargets(source) = 0;
    newTargets(s(t==source)) = 0;
    newTargets(t(source, ~switchEdge)) = 0;
    
    [~, ind] = sort(newTargets, 'descend');
    t(source, switchEdge) = ind(1:nnz(switchEdge));
end

h = graph(s,t);

% a = zeros(N);
% idx = sub2ind(size(a), s(:),t(:));
% a(idx) = 1;
% a = a + a';
% h = graph(a);

end