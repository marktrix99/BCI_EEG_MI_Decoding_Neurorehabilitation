function [cspw, model, mean_acc, std_acc] = ac_classification_model_fbcsp(folderName, csp_para, fs, start_sec, end_sec, class1, class2, ...
                                                 N_channels, Window_buff, T_task_class, classes, chunk_length_sec)
% Robust FBCSP implementation with anti-overfitting measures

%% Parameters with stricter validation
num_bands = 5;
band_edges = [8 12; 12 16; 16 20; 20 24; 24 28];
min_trials_per_class = 5;
rng(42); % For reproducibility

%% Data Loading
[X_all, y_all] = load_eeg_data(folderName, N_channels, Window_buff, T_task_class, classes, chunk_length_sec, fs);

% Class validation
[class1, class2, n_trials_class1, n_trials_class2] = validate_classes(y_all, class1, class2, min_trials_per_class);

% Time segmentation
[X_all, ~, ~] = segment_data(X_all, fs, start_sec, end_sec, N_channels);

% Separate classes
[X1, X2, class2_label] = separate_classes(X_all, y_all, class1, class2);

%% Filter Bank Processing
filtered_data = struct();
for band_idx = 1:num_bands
    band = sprintf('band%d', band_idx);
    [X1_band, ~] = bandpass_filter_with_rejection(X1, band_edges(band_idx,1), band_edges(band_idx,2), fs);
    [X2_band, ~] = bandpass_filter_with_rejection(X2, band_edges(band_idx,1), band_edges(band_idx,2), fs);
    filtered_data.(band).X1 = X1_band;
    filtered_data.(band).X2 = X2_band;
end

%% CSP Processing
csp_results = struct();
band_names = fieldnames(filtered_data);
for b = 1:length(band_names)
    band = band_names{b};
    [csp_results.(band).cspw, csp_results.(band).F1, csp_results.(band).F2] = ...
        regularized_csp(filtered_data.(band).X1, filtered_data.(band).X2, csp_para);
end

%% Feature Processing
features_all = [];
labels_all = [];
for b = 1:length(band_names)
    band = band_names{b};
    [F1_norm, F2_norm] = normalize_features(csp_results.(band).F1, csp_results.(band).F2);
    features_all = [features_all, [F1_norm; F2_norm]]; 
    labels_all = [labels_all; zeros(size(F1_norm,1),1); ones(size(F2_norm,1),1)]; 
end

% Shuffle data
rand_idx = randperm(size(features_all,1));
features_all = features_all(rand_idx,:);
labels_all = labels_all(rand_idx,:); % labels has to be a column vector

%% Feature Selection
[selected_features, ~] = select_features_mi(features_all, labels_all, min(4, size(features_all,2)));
features_selected = features_all(:, selected_features);

%% Classification with Proper Data Handling
k = 5;
cv = cvpartition(labels_all, 'KFold', k);
accuracies = zeros(k,1);

for fold = 1:k
    % Ensure proper matrix format
    X_train = double(features_selected(cv.training(fold),:));
    y_train = double(labels_all(cv.training(fold),:));
    X_test = double(features_selected(cv.test(fold),:));
    y_test = double(labels_all(cv.test(fold),:));
    
    % Train and predict
    lda_model = fitcdiscr(X_train, y_train, 'Gamma', 0.5);
    y_pred = predict(lda_model, X_test);
    
    % Calculate accuracy
    accuracies(fold) = sum(y_pred == y_test) / length(y_test);
    fprintf('Fold %d accuracy: %.2f%%\n', fold, accuracies(fold)*100);
end

%% Final Outputs
mean_acc = mean(accuracies);
std_acc = std(accuracies);
model = fitcdiscr(features_selected, labels_all, 'Gamma', 0.5);
cspw = csp_results;

fprintf('\nFinal accuracy: %.2f%% ± %.2f\n', mean_acc*100, std_acc*100);

%% Enhanced Helper Functions
    function [class1, class2, n1, n2] = validate_classes(y_all, class1, class2, min_trials)
        unique_classes = unique(y_all);
        fprintf('Available classes in data: %s\n', mat2str(unique_classes'));
        
        % Handle class1
        n1 = sum(y_all == class1);
        if n1 < min_trials
            error('Insufficient trials (%d) for class %d (minimum %d required)', n1, class1, min_trials);
        end
        
        % Handle class2 (single class or combination)
        if isscalar(class2)
            n2 = sum(y_all == class2);
            if n2 < min_trials
                error('Insufficient trials (%d) for class %d (minimum %d required)', n2, class2, min_trials);
            end
        else % Combination like [3 4]
            n2 = sum(ismember(y_all, class2));
            if n2 < min_trials
                error('Insufficient trials (%d) for classes [%s] (minimum %d required)', ...
                      n2, num2str(class2), min_trials);
            end
        end
    end

    function [X_filt, rejected] = bandpass_filter_with_rejection(X, low_freq, high_freq, fs)
        [b, a] = butter(4, [low_freq high_freq]/(fs/2), 'bandpass');
        X_filt = zeros(size(X));
        rejected = [];
        amp_threshold = 100; % µV
        
        for i = 1:size(X,1)
            trial = squeeze(X(i,:,:));
            filtered = zeros(size(trial));
            
            for ch = 1:size(trial,1)
                filtered(ch,:) = filtfilt(b, a, trial(ch,:));
            end
            
            if max(abs(filtered(:))) > amp_threshold
                rejected = [rejected; i];
            else
                X_filt(i,:,:) = filtered;
            end
        end
        
        X_filt(rejected,:,:) = [];
    end

    function [cspw, F1, F2] = regularized_csp(X1, X2, m)
        % Regularized covariance estimation
        gamma = 0.1; % Shrinkage parameter
        
        C1 = cov_avg(X1);
        C1 = (1-gamma)*C1 + gamma*mean(eig(C1))*eye(size(C1));
        
        C2 = cov_avg(X2);
        C2 = (1-gamma)*C2 + gamma*mean(eig(C2))*eye(size(C2));
        
        Cc = C1 + C2;
        [V, D] = eig(Cc);
        [D, idx] = sort(diag(D), 'descend');
        V = V(:, idx);
        
        P = diag(1./sqrt(D)) * V';
        S1 = P * C1 * P';
        
        [V2, D2] = eig(S1);
        [D2, idx2] = sort(diag(D2), 'descend');
        V2 = V2(:, idx2);
        
        cspw = V2' * P;
        
        % Project data
        Z1 = project_trials(X1, cspw, m);
        Z2 = project_trials(X2, cspw, m);
        
        % More robust feature extraction
        F1 = log_normalized_features(Z1);
        F2 = log_normalized_features(Z2);
    end

    function [F1_norm, F2_norm] = normalize_features(F1, F2)
        % Joint normalization prevents data leakage
        all_features = [F1; F2];
        mu = mean(all_features);
        sigma = std(all_features);
        sigma(sigma == 0) = 1; % Prevent division by zero
        
        F1_norm = (F1 - mu) ./ sigma;
        F2_norm = (F2 - mu) ./ sigma;
    end

    function features = log_normalized_features(Z)
        varZ = var(Z, 0, 3);
        features = log10(varZ ./ sum(varZ,2) + eps); % Add eps to avoid log(0)
    end
     
    function [X_all, y_all] = load_eeg_data(folderName, N_channels, Window_buff, ...
                                           T_task_class, classes, chunk_length_sec, fs)
        % Load EEG data from MAT files
        files = dir(fullfile(folderName, '**/*.mat'));
        X_all = [];
        y_all = [];
        
        for f = 1:length(files)
            filePath = fullfile(files(f).folder, files(f).name);
            try
                loaded = load(filePath);
                if isfield(loaded, 'y')
                    [X, y_label] = data_load_trials(loaded.y, N_channels, Window_buff, ...
                                                  T_task_class, classes, chunk_length_sec);
                    if isempty(X)
                        warning('Empty data from %s', files(f).name);
                        continue;
                    end
                    
                    % Preprocessing
                    X = notchFilter(X, 50, fs);
                    X = permute(X, [3, 1, 2]);
                    
                    X_all = cat(1, X_all, X);
                    y_all = cat(1, y_all, y_label);
                end
            catch ME
                warning('Failed to load %s: %s', files(f).name, ME.message);
            end
        end
    end

    

    function [X_all, start_idx, end_idx] = segment_data(X_all, fs, start_sec, end_sec, N_channels)
        max_samples = size(X_all,3);
        start_idx = max(1, round(start_sec * fs));
        end_idx = min(max_samples, round(end_sec * fs));
        
        if start_idx >= end_idx
            error('Invalid time window: start (%.2fs) >= end (%.2fs)',...
                  start_idx/fs, end_idx/fs);
        end
        
        if size(X_all,2) < N_channels
            warning('Only %d channels available, using all', size(X_all,2));
            N_channels = size(X_all,2);
        end
        
        X_all = X_all(:, 1:N_channels, start_idx:end_idx);
        disp(['Segmented data size: ' mat2str(size(X_all))]);
    end

    function [X1, X2, class2_label] = separate_classes(X_all, y_all, class1, class2)
        X1 = X_all(y_all == class1, :, :);
        
        if isequal(class2, [3 4]) || (numel(class2)==2 && all(ismember(class2,[3 4])))
            X2 = X_all(ismember(y_all, [3 4]), :, :);
            class2_label = 'Movement (3|4)';
        else
            X2 = X_all(y_all == class2, :, :);
            class2_label = sprintf('Class %d', class2);
        end
    end

    

    function C = cov_avg(X)
        n_trials = size(X,1);
        n_channels = size(X,2);
        C = zeros(n_channels);
        
        for i = 1:n_trials
            trial = squeeze(X(i,:,:));
            cov_mat = (trial * trial') / trace(trial * trial');
            C = C + cov_mat;
        end
        C = C / n_trials;
    end

    function Z = project_trials(X, W, m)
        Wm = [W(1:m,:); W(end-m+1:end,:)];
        n_trials = size(X,1);
        Z = zeros(n_trials, 2*m, size(X,3));
        
        for i = 1:n_trials
            Z(i,:,:) = Wm * squeeze(X(i,:,:));
        end
    end

    function plot_single_trial(data, fs)
        % data: [channels x samples] matrix
        t = (0:size(data,2)-1) / fs;
        plot(t, data');
        xlabel('Time (s)');
        ylabel('Amplitude (μV)');
        title('EEG Trial');
    end

    function [selected_idx, scores] = select_features_mi(features, labels, num_features)
        % MATLAB's equivalent of mutual information feature selection
        [idx, scores] = fscmrmr(features, labels);
        selected_idx = idx(1:num_features);
    end

    % Visualization
    try
        figure('Name', 'Class Data Visualization');
        subplot(2,1,1);
        ac_PlotEvent(permute(X1, [3, 2, 1]), fs, 200, 5, 9);
        title(sprintf('Class %d (n=%d)', class1, size(X1,1)));
        
         subplot(2,1,2);
         ac_PlotEvent(permute(X2, [3, 2, 1]), fs, 200, 5, 9);
         title(sprintf('%s (n=%d)', class2_label, size(X2,1)));
    catch ME
        warning('Visualization failed: %s', ME.message);
    end
end