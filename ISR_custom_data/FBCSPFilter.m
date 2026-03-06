function features = FBCSPFilter(eeg, cspw, csp_para)
% 
% extract features from EEG data using Filter Bank CSP
%
% Input:
%   eeg: EEG signals as [Ns × Nc × Nt] matrix where
%       Ns: number of samples per trial
%       Nc: number of channels (EEG electrodes)
%       Nt: number of trials
%   cspw: FBCSP projection matrices (struct with per band-specific CSP filters)
%   csp_para: number of CSP filter pairs to use per band (will extract 2*csp_para features per band)
%
% Output:
%   features: extracted features as [Nt × (num_bands * 2 * csp_para)] matrix

num_bands = length(fieldnames(cspw)); % Number of frequency bands
nbTrials = size(eeg, 3);
features = zeros(nbTrials, num_bands * 2 * csp_para);

band_names = fieldnames(cspw);
feature_idx = 1;

% same as CSP, but in a loop, per band
for b = 1:num_bands
    band = band_names{b};
    current_cspw = cspw.(band).cspw;
    
    Filter = current_cspw([1:csp_para (end - csp_para + 1):end], :);
    
    % Extract features for each trial in this band
    for t = 1:nbTrials
        % Project the data onto CSP filters
        projectedTrial = Filter * squeeze(eeg(:,:,t))';
        
        % Calculate log-variance features
        variances = var(projectedTrial, 0, 2);
        features(t, feature_idx:feature_idx+2*csp_para-1) = log(1 + variances)';
        end
    
    feature_idx = feature_idx + 2 * csp_para;
    end
end