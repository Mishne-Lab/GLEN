function chicago_component_viz(M, j, figid)
%CHICAGO_COMPONENT_VIZ View single ktensor component for Chicago Crime.
%
%   CHICAGO_COMPONENT_VIZ(M,J,FIGID) plots the Jth component of the ktensor
%   M in figure FIGID.

%% Read in crime type labels
fileID = fopen('modedata/mode-4-primarytype.map','r');
crime_names = textscan(fileID, '%s%[^\n\r]', 'Delimiter', ',', 'MultipleDelimsAsOne', true, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
crime_names = lower(crime_names{1});

%% Read in chicago shape file
S = shaperead('geodata/chicago_geo_data.shp', 'UseGeoCoords', true);


%% Sort out the model tensor
M = normalize(arrange(M),1);
r = ncomponents(M);
U1 = M{1};
U2 = M{2};
U3 = M{3};
U4 = M{4};

% One figure per component
figure(figid); clf;

%% Relative spacings
xx = [1 20 1 20 1 35 1];
xx = xx / sum(xx);
yy = [1 30 3 20 3 ];
yy = yy / sum(yy);

% Create the global axis
GlobalAxis = axes('Position',[0 0 1 1]); % Global Axes
colors = get(gca, 'ColorOrder');
axis off;

% Create and plot dates in upper left 
dateaxes = axesarea(xx(1), sum(yy(1:3)), sum(xx(2:4)), yy(4), 0.05, 0.3);
plot(U1(:,j),'Color', colors(1,:));
ylim([0, max(U1(:))]);
set(dateaxes,'XTick',1:365:6186);
set(dateaxes,'XLim',[0,6186]);
set(dateaxes,'XTickLabel',2001:2018);
set(dateaxes,'XTickLabelRotation',-90);
set(dateaxes,'FontSize',12);
title('Date')

% Create and plot hour of the day in the middle left
houraxes = axesarea(xx(1), yy(1), xx(2), yy(2), 0.2, 0.2);
bar(U2(:,j),'FaceColor',colors(5,:));
set(houraxes,'FontSize',12);
set(houraxes,'XTick',6:6:24);
set(houraxes,'XMinorTick','on');
ylim([0, max(U2(:))]);
set(dateaxes,'XTick',1:365:6186);
title('Hour of Day')

% Create and plot top crimes in lower right
crimeaxes = axesarea(sum(xx(1:3)), yy(1), xx(4), yy(2), 0.4, 0.2);
nn = 5;
[srt, sidx] = sort(U4(:,j),'ascend');
barh(srt(end-nn:end),'FaceColor',colors(7,:));
set(crimeaxes,'YTick',1:nn+1);
set(crimeaxes,'YTickLabel',crime_names(sidx(end-nn:end)));
set(crimeaxes,'FontSize',12);
xlim([0 1]);
title('Top Crimes');

% Create and plot heatmap on entire right side
mapaxes = axesarea(sum(xx(1:5)), yy(1), xx(6),sum(yy(2:4)),0,0);
axes(mapaxes);
set(mapaxes,'FontSize',12);
chicago_heatmap(U3(:,j), min(U3(:)), max(U3(:)), S);
title('Areas')

%%
% Resize 
wpx = 1300;
hpx = 550;
% Resize (keeping position of upper left corner the same)
width = wpx/100;     % Width in inches
height = hpx/100;    % Height in inches
pos = get(gcf, 'Position');
delta_y = max(0, height*100 - pos(4));
set(gcf, 'Position', [pos(1) (pos(2)-delta_y) width*100, height*100]); 
%%

function ax = axesarea(x,y,dx,dy,xop,yop)

xo = xop * dx;
yo = yop * dy;
ax = axes('Position',[x + xo, y + yo, dx - xo, dy - yo]);