function [cspw, model, mean_acc, std_acc] = ac_classification_model_adapted(folderName, csp_para, fs, ...
    start_sec, end_sec, class1, class2)

% ac_classification_model_clean
% Computes CSP + LDA classifier on EEG dataset with preprocessed trials

% Parameters
K = 5;                % K-fold CV
rng(42);              % Reproducibility
N_channels = 16;      % Adjust if needed

% === Initialize ===
X_all = [];
y_all = [];

matFiles = dir(fullfile(folderName, '*.mat'));

for f = 1:length(matFiles)
    filePath = fullfile(folderName, matFiles(f).name);
    data = load(filePath);

    if isfield(data, 'X') && isfield(data, 'y_label')
        X = data.X;
        y = data.y_label;

        % Fix shape if needed
        if ndims(X) == 3 && size(X, 3) == length(y)
            X = permute(X, [3, 1, 2]);  % [trials x channels x samples]
        end

        % Apply filters
        X = notchFilter(X, 50, fs);
        X = BandPass4ndOrderButter(X, 4, 8, 30, fs);

        X_all = cat(1, X_all, X);
        y_all = cat(1, y_all, y);
    else
        warning('Skipping %s: missing X or y_label', filePath);
    end
end

% === Crop Time Window ===
start_idx = round(start_sec * fs);
end_idx = round(end_sec * fs);
X_all = X_all(:, 1:N_channels, start_idx:end_idx);  % [trials x channels x samples]

% === Select Classes ===
X1 = X_all(y_all == class1, :, :);
X2 = X_all(y_all == class2, :, :);
y1 = zeros(size(X1, 1), 1);
y2 = ones(size(X2, 1), 1);

% === K-Fold Evaluation ===
cv1 = cvpartition(size(X1,1), 'KFold', K);
cv2 = cvpartition(size(X2,1), 'KFold', K);

accuracies = zeros(K, 1);

for k = 1:K
    % Split
    X1_tr = X1(training(cv1, k), :, :); y1_tr = y1(training(cv1, k));
    X1_te = X1(test(cv1, k), :, :);     y1_te = y1(test(cv1, k));
    
    X2_tr = X2(training(cv2, k), :, :); y2_tr = y2(training(cv2, k));
    X2_te = X2(test(cv2, k), :, :);     y2_te = y2(test(cv2, k));

    % Train CSP
    [W, ~] = compute_csp(X1_tr, X2_tr);
    Wm = [W(1:csp_para, :); W(end-csp_para+1:end, :)];

    % Train features
    F_tr = extract_features(project_trials(cat(1, X1_tr, X2_tr), Wm));
    y_tr = [y1_tr; y2_tr];

    model = fitcdiscr(F_tr, y_tr);  % LDA

    % Test
    F_te = extract_features(project_trials(cat(1, X1_te, X2_te), Wm));
    y_te = [y1_te; y2_te];
    y_pred = predict(model, F_te);

    accuracies(k) = mean(y_pred == y_te);
    fprintf('Fold %d: Accuracy = %.2f%%\n', k, accuracies(k) * 100);
end

% Final CSP from all data
[cspw, ~] = compute_csp(X1, X2);

% Final LDA
Z_all = project_trials(cat(1, X1, X2), [cspw(1:csp_para, :); cspw(end-csp_para+1:end, :)]);
F_all = extract_features(Z_all);
y_feat = [zeros(size(X1, 1), 1); ones(size(X2, 1), 1)];
model = fitcdiscr(F_all, y_feat);

% Report
mean_acc = mean(accuracies);
std_acc = std(accuracies);
fprintf('Mean Accuracy: %.2f%%\n', mean_acc * 100);
fprintf('Std Accuracy:  %.2f%%\n', std_acc * 100);

end

%% === Helper Functions ===
function [W, P] = compute_csp(X1, X2)
    C1 = cov_avg(X1);
    C2 = cov_avg(X2);
    [V, D] = eig(C1 + C2);
    P = sqrtm(inv(D)) * V';
    S1 = P * C1 * P';
    [B, ~] = eig(S1);
    W = B' * P;
end

function C = cov_avg(X)
    n = size(X, 1); C = 0;
    for i = 1:n
        x = squeeze(X(i, :, :));
        c = (x * x') / trace(x * x');
        C = C + c;
    end
    C = C / n;
end

function Z = project_trials(X, W)
    n = size(X, 1);
    m = size(W, 1);
    Z = zeros(n, m, size(X, 3));
    for i = 1:n
        Z(i, :, :) = W * squeeze(X(i, :, :));
    end
end

function F = extract_features(Z)
    n = size(Z, 1);
    F = zeros(n, size(Z, 2));
    for i = 1:n
        varZ = var(squeeze(Z(i, :, :))');
        F(i, :) = log10(varZ / sum(varZ));
    end
end
