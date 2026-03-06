% === Constants ===
N_channels       = 16;
Window_buff      = 256;
T_task_class     = [6, 2, 6, 6, 90];  % durations for classes
classes          = [1, 2, 3, 4, 5];   % labels (rest, relax, closed, open, long rest)
chunk_length_sec = 6;
start_sec        = 0.5;
end_sec          = 2.5;
movement_comb    = [3 4];
fs               = 256;
filter_list      = [2, 4];

% === Participant folder ===
participantFolder = 'C:\Users\LENOVO\Documents\Uni\Master Thesis\ISR_Data\Test_31_07_2025\Calibração\Giovanni_24_31_07_real_movement';

% --- Automatically detect all subfolders containing any signals.mat files ---
allSubfolders = dir(participantFolder);
allSubfolders = allSubfolders([allSubfolders.isdir]); % only directories
sessionFolders = {};

for k = 1:length(allSubfolders)
    folderPath = fullfile(allSubfolders(k).folder, allSubfolders(k).name);
    if ~ismember(allSubfolders(k).name,{'.','..'})
        matFiles = dir(fullfile(folderPath, 'signals.mat')); % any signals.mat file
        if ~isempty(matFiles)
            sessionFolders{end+1} = folderPath; %#ok<SAGROW>
            % Display each file found
            for f = 1:length(matFiles)
                fprintf('  Folder: %s | File: %s\n', folderPath, matFiles(f).name);
            end
        end
    end
end

if isempty(sessionFolders)
    error('No session folders with .mat files found in: %s', participantFolder);
end

% === Preallocate results ===
results = [];

% --- Loop over filter settings ---
for i = 1:length(filter_list)
    csp_para = filter_list(i);

    % --- Merge all sessions data into X_all and y_all ---
    X_all = [];
    y_all = [];
    for s = 1:length(sessionFolders)
        folderName = sessionFolders{s};
        [X_sess, y_sess] = load_eeg_data(folderName, N_channels, Window_buff, ...
                                         T_task_class, classes, chunk_length_sec, fs);
        X_all = cat(1, X_all, X_sess);
        y_all = cat(1, y_all, y_sess);
    end

    % --- Run models for each comparison ---
    [~, ~, acc_closed, std_closed]     = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, 1, 3, N_channels, Window_buff, T_task_class, classes, chunk_length_sec);
    [~, ~, acc_opened, std_opened]     = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, 1, 4, N_channels, Window_buff, T_task_class, classes, chunk_length_sec);
    [~, ~, acc_movement, std_movement] = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, 1, movement_comb, N_channels, Window_buff, T_task_class, classes, chunk_length_sec);

    % --- Create table after variables exist ---
    tmp_table = table( ...
        repmat(csp_para, 3, 1), ...
        ["Rest vs Closed"; "Rest vs Opened"; "Rest vs Movement"], ...
        [acc_closed; acc_opened; acc_movement], ...
        [std_closed; std_opened; std_movement], ...
        'VariableNames', {'CSPFilters', 'Comparison', 'MeanAccuracy', 'StdAccuracy'} ...
    );

    results = [results; tmp_table];
end

%% === Compute overall accuracy per CSP filter ===
filterListUnique = unique(results.CSPFilters(~isnan(results.CSPFilters)));

for i = 1:length(filterListUnique)
    filt = filterListUnique(i);
    idx = results.CSPFilters == filt;
    acc = results.MeanAccuracy(idx);
    std_acc = results.StdAccuracy(idx);
    
    overallAcc = mean(acc)*100;
    overallStd = std(acc)*100;

    fprintf('\n=== Overall Accuracy for CSP Filter %d ===\n', filt);
    fprintf('Mean Accuracy: %.2f%% ± %.2f%%\n', overallAcc, overallStd);
    
    %add as row to the results table
    overallRow = table( ...
        filt, ...
        "Overall Accuracy", ...
        overallAcc, ...
        overallStd, ...
        'VariableNames', {'CSPFilters', 'Comparison', 'MeanAccuracy', 'StdAccuracy'} ...
    );
    
    results = [results; overallRow];
end

% --- Define folder for saving results ---
resultsFolder = 'C:\Users\LENOVO\Documents\Uni\Master Thesis\ISR_Data\Overall_Results_Calibracao';
% --- Save updated table with per-filter overall accuracy ---
outputFile = fullfile(resultsFolder, 'Giovanni_24_31_07_real_movement_results.xlsx');
writetable(results, outputFile);
disp(results);
    
