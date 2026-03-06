function features = CSPFilter(eeg,cspw,csp_para)
%
%   extract features from an EEG data set using the Common Spatial Patterns (CSP) algorithm
%
%  Input:
%   eeg: the EEGSignals from which extracting the CSP features. These signals
%   are a structure such that:
%   eeg: the EEG signals as a [Ns * Nc * Nt] Matrix where
%       Ns: number of EEG samples per trial
%       Nc: number of channels (EEG electrodes)
%       nT: number of trials
%   fbcspw: the CSP projection matrix, learnt previously (see function learnCSP)
%   filterorder: number of pairs of CSP filters to be used. The number of
%   features extracted will be twice the value of this parameter. The
%   filters selected are the one corresponding to the lowest and highest
%   eigenvalues
%
%  Output:
%  features: the features extracted from this EEG data set 
%   as a [Nt * (nbFilterPairs*2 + 1)] matrix, with the class labels as the
%   last column   


nbTrials = size(eeg,3);
features = zeros(nbTrials, 2 * csp_para + 1);
Filter = cspw([1:csp_para (end - csp_para + 1) : end], :);

%extracting the CSP features from each trial
for t = 1 : nbTrials    
    %projecting the data onto the CSP filters    
    projectedTrial = Filter * eeg(:,:,t)';    

    %generating the features as the log variance of the projectedsignals

    variances = var(projectedTrial,0,2);    
    for f = 1 : length(variances)
        features(t,f) = log(1 + variances(f));
    end
end