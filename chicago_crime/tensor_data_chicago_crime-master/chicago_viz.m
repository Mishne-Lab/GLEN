function chicago_viz(M,figid,showzero)
%CHICAGO_VIZ View ktensor of Chicago Crime tensor.
%
%   CHICAGO_VIZ(M,FIGID) plots the CP decomposition of the Chicago Crime
%   tensor in the figure FIGID. Note this shows only the top-13 most
%   prevalent crimes in the 4th mode.
%
%%
if ~exist('showzero','var')
    showzero = false(4,1);
end
%%

r = ncomponents(M);

%% Read in crime type labels
fileID = fopen('modedata/mode-4-primarytype.map','r');
crime_names = textscan(fileID, '%s%[^\n\r]', 'Delimiter', ',', 'MultipleDelimsAsOne', true, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
crime_names = lower(crime_names{1});

%% Read in crime frequencies for sorting, saved special
% This file was created via crime_freq = collapse(Xbinary,-4) on the
% binarized tensor.
load('modedata/crime_freq')
[scf, sidx] = sort(crime_freq, 'descend');

%% Visuzalize results
vh = viz(M, 'Figure',figid,...
    'PlotCommands',{'plot',@(x,y) bar(x,y,'r'),...
    @(x,y) bar(x,y,'k'),@(x,y) bar(x,y(sidx),'g')},...
    'Ylims',{[0,17],'same','same','same'},...
    'ShowZero',showzero,...
    'ModeTitles',{'Date (Tick = 1 Year)','Hour','Neighborhood','Crime'},...
    'RelModeWidth',[2 0.5 0.5 1],...
    'Normalize',@(X) normalize(arrange(X),1),...
    'FactorTitles','number',...
    'LeftSpace',0.025,'BottomSpace',0.2);

%% Good size for Powerpoint slide
wpx = 1300;
hpx = 550;
% Resize (keeping position of upper left corner the same)
width = wpx/100;     % Width in inches
height = hpx/100;    % Height in inches
pos = get(gcf, 'Position');
delta_y = max(0, height*100 - pos(4));
set(gcf, 'Position', [pos(1) (pos(2)-delta_y) width*100, height*100]); 

%% Fix up Mode 1 - show year markings
for i = 1:r
    set(vh.FactorAxes(1,i),'XTick',1:365:6186);
end
set(vh.FactorAxes(1,r),'XTickLabel',2001:2018);
set(vh.FactorAxes(1,r),'XTickLabelRotation',-90);

%% Fix up Mode 2 - show 6-hour markings and lengthen
yl = get(vh.FactorAxes(2,1),'YLim');
for i = 1:r
    set(vh.FactorAxes(2,i),'TickLength',[0.0500 0.20]);
    set(vh.FactorAxes(2,i),'XTick',(1:6:25)-0.5);
    %set(vh.FactorAxes(2,i),'XMinorTick','on');
end
set(vh.FactorAxes(2,r),'XTickLabel',(1:6:25)-1);
set(vh.FactorAxes(2,r),'XTickLabelRotation',-90);

%% Fix up Mode 3
for i = 1:r
    set(vh.FactorAxes(3,i),'XTick',11:11:77);
end
set(vh.FactorAxes(3,r),'XTickLabelRotation',-90);

%% Fix up Mode 4 - show crime type labels
for i = 1:r
    set(vh.FactorAxes(4,i),'XTick',1:13);
    set(vh.FactorAxes(4,i),'XLim',[0.5 13.5]);
end
set(vh.FactorAxes(4,r),'XTickLabel',crime_names(sidx));
set(vh.FactorAxes(4,r),'XTickLabelRotation',-90);
