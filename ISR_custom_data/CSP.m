function [cspfeature, cspw] = CSP(eeg,label,m)
% Common Spatial Pattern
% Input:
% eeg: Raw EEG Data, the data format is [Ntime*Nchans*Ntrial], where Nchans
%   is number of channels (EEG electrodes) and the Ntime is the number of
%   EEG samples, and Ntrial is the number of trials.
% label£∫data label, the value only is 1and 2
% m£∫csp para
% Output:
% cspfeature:
% cspw:fbcsp 

%check and initializations
EEG_Channels = size(eeg,2);
EEG_Trials = size(eeg,3); 
classLabels = unique(label);% Return non-repeating values
EEG_Classes = length(classLabels);

covMatrix = cell(EEG_Classes,1);
% Computing the normalized covariance matrices for each trial
trialCov = zeros(EEG_Channels,EEG_Channels,EEG_Trials);
for i = 1:EEG_Trials
    E = eeg(:,:,i)';
    EE = E*E';
    trialCov(:,:,i) = EE./trace(EE);  % º∆À„–≠∑Ω≤Óæÿ’Û
end
clear E;
clear EE;

for i = 1:EEG_Classes
    covMatrix{i} = mean(trialCov(:,:,label == classLabels(i)),3);
end

covTotal = covMatrix{1} + covMatrix{2};

[Uc,Dt] = eig(covTotal);

eigenvalues = diag(Dt);
[eigenvalues,egIndex] = sort(eigenvalues, 'descend');
Ut = Uc(:,egIndex);

P = diag(sqrt(1./eigenvalues))*Ut';

transformedCov1 = P*covMatrix{1}*P';

[U1,D1] = eig(transformedCov1);
eigenvalues = diag(D1);
[eigenvalues,egIndex] = sort(eigenvalues, 'descend');
U1 = U1(:, egIndex);

CSPMatrix = U1' * P;

FilterPairs = m;     
features_train = zeros(EEG_Trials, 2*FilterPairs+1);

Filter = CSPMatrix([1:FilterPairs (end-FilterPairs+1):end],:);
cspw = Filter;

%extracting the CSP features from each trial
for t=1:EEG_Trials    
    % projecting the data onto the CSP filters    
    projectedTrial_train = Filter * eeg(:,:,t)';    
    
    % generating the features as the log variance of the projected signals
    variances_train = var(projectedTrial_train,0,2);  
    
    for f=1:length(variances_train)
        features_train(t,f) = log(variances_train(f));
        % features_train(t,f) = log(variances_train(f)/sum(variances_train));  
    end
    
end
cspfeature = features_train(:,1:2*m);

end
