function [cspw, model, mean_acc, std_acc] = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, class1, class2,...
                                                 N_channels, Window_buff, T_task_class, classes, chunk_length_sec)
% Enhanced CSP-LDA classifier with robust error handling

%% === Input Validation ===
if ~isfolder(folderName)
    error('Folder not found: %s', folderName);
end

if ~all(ismember([class1, class2], [1,3,4,5]))
    error('Invalid class labels. Only 1 (rest), 3 (closed), or 4 (opened) allowed');
end

%% === Data Loading with Debugging ===
disp(['=== Loading data from: ' folderName ' ===']);
[X_all, y_all] = load_eeg_data(folderName, N_channels, Window_buff,...
                              T_task_class, classes, chunk_length_sec, fs);

%% === Data Validation ===
if isempty(X_all)
    error('No valid EEG data loaded. Check:\n1. MAT files exist\n2. Files contain ''y'' variable\n3. data_load_trials() works');
end

disp(['Loaded ' num2str(size(X_all,1)) ' trials']);
% disp('Class distribution:');
tabulate(y_all);

%% === Flexible Class Handling ===
[class1, class2] = validate_classes(y_all, class1, class2);

%% === Safe Data Segmentation ===
[X_all, start_idx, end_idx] = segment_data(X_all, fs, start_sec, end_sec, N_channels);

%% === Class Separation ===
[X1, X2] = separate_classes(X_all, y_all, class1, class2);

%% === CSP Feature Extraction ===
[cspw, F1, F2] = extract_csp_features(X1, X2, csp_para);

%% === Classification Pipeline ===
[model, mean_acc, std_acc] = run_classification(X1, X2, csp_para);

%% === Helper Functions ===

function [X_all, y_all] = load_eeg_data(folderName, N_channels, Window_buff,...
                                       T_task_class, classes, chunk_length_sec, fs)
    % Recursively load all MAT files in folder
    X_all = [];
    y_all = [];
    files = dir(fullfile(folderName, '**/*.mat')); % Search recursively
    
    if isempty(files)
        error('No MAT files found in: %s', folderName);
    end
    
    for f = 1:length(files)
        filePath = fullfile(files(f).folder, files(f).name);
        try
            loaded = load(filePath);
            if isfield(loaded, 'y')
                [X, y_label] = data_load_trials(loaded.y, N_channels, Window_buff,...
                                              T_task_class, classes, chunk_length_sec);
                if isempty(X)
                    warning('Empty data from %s', files(f).name);
                    continue;
                end
                
                % Preprocessing
                X = notchFilter(X, 50, fs);
                X = BandPass4ndOrderButter(X, 4, 8, 30, fs);
                X = permute(X, [3, 1, 2]);
                
                X_all = cat(1, X_all, X);
                y_all = cat(1, y_all, y_label);
            end
        catch ME
            warning('Failed to load %s: %s', files(f).name, ME.message);
        end
    end

    y_all(y_all == 5) = 1;
end

function [class1, class2] = validate_classes(y_all, class1, class2)
    % Ensure requested classes exist in data
    unique_classes = unique(y_all);
    
    if ~ismember(class1, unique_classes)
        error('Class %d not found in data. Available classes: %s',...
              class1, mat2str(unique_classes'));
    end
    
    % Handle movement class case (3|4)
    if isequal(class2, [3 4]) || (numel(class2)==2 && all(ismember(class2,[3 4])))
        if ~any(ismember([3 4], unique_classes))
            error('No movement classes (3 or 4) found in data');
        end
    elseif ~ismember(class2, unique_classes)
        available = setdiff(unique_classes, class1);
        if isempty(available)
            error('No valid class2 available');
        else
            class2 = available(1);
            warning('Using class %d instead of specified class2', class2);
        end
    end
end

function [X_all, start_idx, end_idx] = segment_data(X_all, fs, start_sec, end_sec, N_channels)
    % Safe time window segmentation
    max_samples = size(X_all,3);
    start_idx = max(1, round(start_sec * fs));
    end_idx = min(max_samples, round(end_sec * fs));
    
    if start_idx >= end_idx
        error('Invalid time window: start (%.2fs) >= end (%.2fs)',...
              start_idx/fs, end_idx/fs);
    end
    
    % Channel selection
    if size(X_all,2) < N_channels
        warning('Only %d channels available, using all', size(X_all,2));
        N_channels = size(X_all,2);
    end
    
    X_all = X_all(:, 1:N_channels, start_idx:end_idx);
    % disp(['Segmented data size: ' mat2str(size(X_all))]);
end

function [X1, X2] = separate_classes(X_all, y_all, class1, class2)
    % Separate classes with visualization
    X1 = X_all(y_all == class1, :, :);
    
    if isequal(class2, [3 4]) || (numel(class2)==2 && all(ismember(class2,[3 4])))
        X2 = X_all(ismember(y_all, [3 4]), :, :);
        class2_label = 'Movement (3|4)';
    else
        X2 = X_all(y_all == class2, :, :);
        class2_label = sprintf('Class %d', class2);
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
    
    if isempty(X2)
        error('No trials found for class2');
    end
end

function [cspw, F1, F2] = extract_csp_features(X1, X2, csp_para)
    % Compute CSP features
    try
        [cspw, ~] = compute_csp(X1, X2);
        Z1 = project_trials(X1, cspw, csp_para);
        Z2 = project_trials(X2, cspw, csp_para);
        F1 = extract_features(Z1);
        F2 = extract_features(Z2);
    catch ME
        error('Feature extraction failed: %s', ME.message);
    end
end

function [model, mean_acc, std_acc] = run_classification(X1, X2, csp_para)
    % K-fold cross validation
    rng(42); % For reproducibility
    K = 5;
    cv1 = cvpartition(size(X1,1), 'KFold', K);
    cv2 = cvpartition(size(X2,1), 'KFold', K);
    accuracies = zeros(K,1);
    
    for fold = 1:K
        % Split data
        X1_train = X1(training(cv1,fold),:,:);
        X1_test = X1(test(cv1,fold),:,:);
        X2_train = X2(training(cv2,fold),:,:);
        X2_test = X2(test(cv2,fold),:,:);
        
        % Combine and label
        X_train = cat(1, X1_train, X2_train);
        y_train = [zeros(size(X1_train,1),1); ones(size(X2_train,1),1)];
        X_test = cat(1, X1_test, X2_test);
        y_test = [zeros(size(X1_test,1),1); ones(size(X2_test,1),1)];
        
        % Fold-specific processing
        [cspw_fold, ~] = compute_csp(X1_train, X2_train);
        Z_train = project_trials(X_train, cspw_fold, csp_para);
        F_train = extract_features(Z_train);
        
        Z_test = project_trials(X_test, cspw_fold, csp_para);
        F_test = extract_features(Z_test);
        
        % Train and evaluate
        model_fold = fitcdiscr(F_train, y_train);
        y_pred = predict(model_fold, F_test);
        accuracies(fold) = mean(y_pred == y_test);
        % fprintf('Fold %d accuracy: %.2f%%\n', fold, accuracies(fold)*100);
    end
    
    % Final model on all data
    Z_all = project_trials(cat(1,X1,X2), cspw, csp_para);
    F_all = extract_features(Z_all);
    y_all = [zeros(size(X1,1),1); ones(size(X2,1),1)];
    model = fitcdiscr(F_all, y_all);
    
    % Return performance metrics
    mean_acc = mean(accuracies);
    std_acc = std(accuracies);
    fprintf('\n=== Final Results ===\n');
    fprintf('Mean accuracy: %.2f%% ± %.2f\n', mean_acc*100, std_acc*100);
end

%% === Core CSP Functions (unchanged from original) ===
function [X_tr, X_te] = split_trials(X, pct)
    n = size(X,1);
    idx = randperm(n);
    n_tr = round(pct * n);
    X_tr = X(idx(1:n_tr),:,:);
    X_te = X(idx(n_tr+1:end),:,:);
end

function [W, P] = compute_csp(X1, X2)
    C1 = cov_avg(X1);
    C2 = cov_avg(X2);
    Cc = C1 + C2;
    [V, D] = eig(Cc);
    P = sqrtm(inv(D)) * V';
    S1 = P * C1 * P';
    [B, ~] = eig(S1);
    W = B' * P;
end

function C = cov_avg(X)
    [n, ch, ~] = size(X);
    C = zeros(ch);
    for i = 1:n
        trial = squeeze(X(i,:,:));
        c = (trial * trial') / trace(trial * trial');
        C = C + c;
    end
    C = C / n;
end

function Z = project_trials(X, W, m)
    Wm = [W(1:m,:); W(end-m+1:end,:)];
    n = size(X,1);
    Z = zeros(n, 2*m, size(X,3));
    for i = 1:n
        Z(i,:,:) = Wm * squeeze(X(i,:,:));
    end
end

function feat = extract_features(Z)
    n = size(Z,1);
    feat = zeros(n, size(Z,2));
    for i = 1:n
        varZ = var(squeeze(Z(i,:,:))');
        feat(i,:) = log10(varZ / sum(varZ));
    end
end

end