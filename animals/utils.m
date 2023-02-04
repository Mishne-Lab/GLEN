% L = Laplacian;
% L = Lcell{3,1};
L = L / trace(L) * 33;
A = -L+diag(diag(L));
g = graph(A);
plot(g,'LineWidth',g.Edges.Weight*10,'NodeColor',"#A2142F",'EdgeColor',"#A2142F",'Layout','circle','NodeLabel',names,'NodeFontSize',10);
axis square;
axis off;
ax=gca;
exportgraphics(ax,'./results/egilmez_cgl_reg2_alpha002.jpg','Resolution',300)
% exportgraphics(ax,'./results/gl_bernoulli_full_animal_graph_alpha-04_beta05.jpg','Resolution',300)