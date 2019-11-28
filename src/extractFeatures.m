%% Add voicebox and MSR Identity toolkits to path
% Voicebox toolkit is needed for: Voice Activity Detection
% MSR Identity toolkit is needed for Extracting MFCC and LPC Features
% Download MSR Identity toolkit from: https://www.microsoft.com/en-us/download/confirmation.aspx?id=52279
% Download Voicebox toolkit from: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
% Place both the toolkits in 'utils' directory

addpath(genpath('utils/MSR Identity Toolkit v1.0'));
addpath(genpath('utils/voicebox'));


%% Path to audio files

audio_filepath_1 = 'sample_audio/1066/12.wav';  % Speaker_ID = 1066
audio_filepath_2 = 'sample_audio/1066/17.wav';  % Speaker_ID = 1066
audio_filepath_3 = 'sample_audio/1055/22.wav';  % Speaker_ID = 1055
audio_filepath_4 = 'sample_audio/1055/1.wav';  % Speaker_ID = 1055
audio_filepath_5 = 'sample_audio/100962/1.wav';  % Speaker_ID = 100962
audio_filepath_6 = 'sample_audio/100962/2.wav';  % Speaker_ID = 100962

%% Path to output feature files
feature_filepath_1 = 'sample_feature/1066/12.mat';  % Speaker_ID = 1066
feature_filepath_2 = 'sample_feature/1066/17.mat';  % Speaker_ID = 1066
feature_filepath_3 = 'sample_feature/1055/22.mat';  % Speaker_ID = 1055
feature_filepath_4 = 'sample_feature/1055/1.mat';  % Speaker_ID = 1055
feature_filepath_5 = 'sample_feature/100962/1.mat';  % Speaker_ID = 100962
feature_filepath_6 = 'sample_feature/100962/2.mat';  % Speaker_ID = 100962


%% Extract MFCC-LPC feature

[feature_1,~] = get_LPC_MFC_feature(audio_filepath_1);
[feature_2,~] = get_LPC_MFC_feature(audio_filepath_2);
[feature_3,~] = get_LPC_MFC_feature(audio_filepath_3);
[feature_4,~] = get_LPC_MFC_feature(audio_filepath_4);
[feature_5,~] = get_LPC_MFC_feature(audio_filepath_5);
[feature_6,~] = get_LPC_MFC_feature(audio_filepath_6);


%% Save Extracted features to file
data = feature_1;
save(feature_filepath_1,'data','-v7.3');

data = feature_2;
save(feature_filepath_2,'data','-v7.3');

data = feature_3;
save(feature_filepath_3,'data','-v7.3');

data = feature_4;
save(feature_filepath_4,'data','-v7.3');

data = feature_5;
save(feature_filepath_5,'data','-v7.3');

data = feature_6;
save(feature_filepath_6,'data','-v7.3');

%% Now open main.py to perform speaker recognition




