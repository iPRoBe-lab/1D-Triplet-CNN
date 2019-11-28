function [ feature, w ] = get_LPC_MFC_feature(audio_path)


%% Read audio from file
[s, fs] = audioread(audio_path);
if ~isempty(s)
    
    %% Perform voice activity detection (VAD) before feature extraction
    l1 = length(s);
    [VS,~]=vadsohn(s,fs);
    s = s(1:length(VS));
    s = s(VS~=0);
    l2 = length(s);
    w = l2/l1;
    
    t = floor([0.01 0.02]*fs);
    %% preemphasis zero is at 50 Hz
    spp=filter([1 -exp(-2*pi*50/fs)],1,s);
    if(length(spp)>t(2))
        mfcc_feat = melcepst(spp,fs,'0d',19,floor(3*log(fs)),t(2));
        mfcc_feat = cmvn(mfcc_feat' ,true);
        lpcc_feat = lpcc(spp,fs,'d',20,floor(3*log(fs)),t(2),t(1));
        lpcc_feat = cmvn(lpcc_feat' ,true);
        feature = single(cat(3,mfcc_feat,lpcc_feat));
    else
        feature = [];
    end
    
    if(isnan(sum(sum(sum(feature)))) || isinf(sum(sum(sum(feature)))))
        disp('Data Corrupt!');
        feature = [];
    end
else
    disp(['Empty audio !  ',audioPath])
    feature = []    ;
end
end

