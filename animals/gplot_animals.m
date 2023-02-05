function gplot_animals(L, names, edge_weight, font_size, graph_color, save)
if nargin < 6
    save = "";
    if nargin < 5
        graph_color = "#A2142F";
        if nargin < 4
            font_size = 10;
            if nargin < 3
                edge_weight = 10;
            end
        end
    end
end

N = size(L,1);
L = L / trace(L) * N;
A = -L+diag(diag(L));
g = graph(A);
plot(g,'LineWidth',g.Edges.Weight*edge_weight,'NodeColor',graph_color,'EdgeColor',graph_color,'Layout','circle','NodeLabel',names,'NodeFontSize',font_size);
axis square;
axis off;
ax=gca;

if save ~= ""
    try
        exportgraphics(ax,save,'Resolution',300);
    catch
        warning("Invalid saving directory! Figures not saved!");
    end
end
end