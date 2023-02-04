%% Conversion Script for Chicago Crime Data
% 
% Requires the following files be downloaded and uncompressed from
% http://frostt.io/tensors/chicago-crime/
%
% * chicago-crime-comm.tns 
% * mode-1-date.map
% * mode-2-time.map
% * mode-3-communityarea.map
% * mode-4-primarytype.map

%% Main tensor
rawdata = dlmread('chicago-crime-comm.tns');
crime_tensor = sptensor(rawdata(:,1:4),rawdata(:,5));

%% Mode 1
fid = fopen('mode-1-date.map');
rawdata = textscan(fid, '%s');
fclose(fid);
crime_mode_1_date = rawdata{1};

%% Mode 2
crime_mode_2_hour = dlmread('mode-2-time.map');

%% Mode 3
[fid] = fopen('mode-3-communityarea.map');
rawdata = textscan(fid, '%s','Delimiter','\n');
fclose(fid);
crime_mode_3_area = rawdata{1};

%% Mode 4
[fid] = fopen('mode-4-primarytype.map');
rawdata = textscan(fid, '%s','Delimiter','\n');
fclose(fid);
crime_mode_4_type = rawdata{1};

%% Save into mat file
save('chicago_crime.mat','crime_tensor','crime_mode_1_date',...
    'crime_mode_2_hour','crime_mode_3_area','crime_mode_4_type');