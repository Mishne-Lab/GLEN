function S = chicago_heatmap(w, cmin, cmax, S)
%CHICAGO_HEATMAP Plot heatmap of Chicago.
%
%  CHICAGO_HEATMAP(W,CMIN,CMAX) plots a heatmap of the 77 Chicago areas
%  using the data in W, where the heatmap has the lower and upper bounds of
%  CMIN and CMAX, respectively. 


%% Read in the data if it doesn't already exist
if ~exist('S','var')
    S = shaperead('geodata/chicago_geo_data.shp', 'UseGeoCoords', true);
end

%% Create the figure and draw chicago
axesm mercator
geoshow(S);

%% Create and apply the color coding

%c = colormap('parula');
c = colormap('hot');
c = colormap(flipud(c));
ncolors = size(c,1);
crange = cmax - cmin;
colorbar
caxis([cmin cmax]);


if max(w) > cmax || min(w) < cmin
    warning('The range for the colormap is not the same as that for the data');
    fprintf('max(w) = %e > cmax = %e or min(w) = %e < cmin = %e\n', max(w), cmax, min(w), cmin);
end

wmap = ((w - cmin) / crange) * ncolors;
wmap = round(wmap);
wmap = min(wmap, ncolors);
wmap = max(wmap, 1);

for i = 1:length(w)
    geoshow(S(i), 'FaceColor', c(wmap(i),:));
end

