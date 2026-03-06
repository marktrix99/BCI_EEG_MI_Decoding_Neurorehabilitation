% === Constants ===
N_channels       = 11;
Window_buff      = 256;
T_task_class     = [6, 2, 6, 6, 90];  % durations for classes
classes          = [1, 2, 3, 4, 5];   % labels (rest, relax, closed, open, long rest)
chunk_length_sec = 6;
start_sec        = 0.5;
end_sec          = 2.5;
movement_comb    = [3 4];
fs               = 256;
filter_list      = [2, 4];

% === Base directory configuration ===
baseDirectory = 'C:\Users\LENOVO\Documents\Uni\Master Thesis\ISR_Data';
calibrationDate = 'Test_23_07_2025'; % 
resultsFolder = fullfile(baseDirectory, 'Overall_Results_Calibracao');

% Create results folder if it doesn't exist
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% === Automatically find all protocol folders ===
calibrationPath = fullfile(baseDirectory, calibrationDate, 'Calibração');
protocolFolders = dir(calibrationPath);
protocolFolders = protocolFolders([protocolFolders.isdir]); % only directories
protocolFolders = protocolFolders(~ismember({protocolFolders.name}, {'.', '..'})); % exclude . and ..

if isempty(protocolFolders)
    error('No protocol folders found in: %s', calibrationPath);
end 

fprintf('Found %d protocol folders:\n', length(protocolFolders));
for i = 1:length(protocolFolders)
    fprintf('  %d: %s\n', i, protocolFolders(i).name);
end

% === Process each protocol folder ===
for protocolIdx = 1:length(protocolFolders)
    protocolName = protocolFolders(protocolIdx).name;
    participantFolder = fullfile(calibrationPath, protocolName);
    
    fprintf('\n=== Processing Protocol: %s ===\n', protocolName);
    
    % --- Automatically detect all subfolders containing signals.mat files ---
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
        fprintf('Warning: No session folders with .mat files found in: %s\n', participantFolder);
        continue; % Skip this protocol and move to next
    end

    % === Preallocate results for this protocol ===
    protocolResults = [];

    % --- Loop over filter settings ---
    for i = 1:length(filter_list)
        csp_para = filter_list(i);

        % --- Merge all sessions data into X_all and y_all ---
        X_all = [];
        y_all = [];
        for s = 1:length(sessionFolders)
            folderName = sessionFolders{s};
            try
                [X_sess, y_sess] = load_eeg_data(folderName, N_channels, Window_buff, ...
                                                 T_task_class, classes, chunk_length_sec, fs);
                X_all = cat(1, X_all, X_sess);
                y_all = cat(1, y_all, y_sess);
            catch ME
                fprintf('Error loading data from %s: %s\n', folderName, ME.message);
                continue;
            end
        end

        if isempty(X_all)
            fprintf('Warning: No data loaded for protocol %s, filter %d\n', protocolName, csp_para);
            continue;
        end

        % --- Run models for each comparison ---
        try
            [~, ~, acc_closed, std_closed]     = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, 1, 3, N_channels, Window_buff, T_task_class, classes, chunk_length_sec);
            [~, ~, acc_opened, std_opened]     = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, 1, 4, N_channels, Window_buff, T_task_class, classes, chunk_length_sec);
            [~, ~, acc_movement, std_movement] = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, 1, movement_comb, N_channels, Window_buff, T_task_class, classes, chunk_length_sec);

            % --- Create table after variables exist ---
            tmp_table = table( ...
                repmat(string(protocolName), 3, 1), ...
                repmat(csp_para, 3, 1), ...
                ["Rest vs Closed"; "Rest vs Opened"; "Rest vs Movement"], ...
                [acc_closed; acc_opened; acc_movement], ...
                [std_closed; std_opened; std_movement], ...
                'VariableNames', {'Protocol', 'CSPFilters', 'Comparison', 'MeanAccuracy', 'StdAccuracy'} ...
            );

            protocolResults = [protocolResults; tmp_table];
            
        catch ME
            fprintf('Error in classification for protocol %s, filter %d: %s\n', protocolName, csp_para, ME.message);
            continue;
        end
    end

    % === Compute overall accuracy per CSP filter for this protocol ===
    if ~isempty(protocolResults)
        filterListUnique = unique(protocolResults.CSPFilters(~isnan(protocolResults.CSPFilters)));

        for i = 1:length(filterListUnique)
            filt = filterListUnique(i);
            idx = protocolResults.CSPFilters == filt;
            acc = protocolResults.MeanAccuracy(idx);
            std_acc = protocolResults.StdAccuracy(idx);
            
            overallAcc = mean(acc)*100;
            overallStd = std(acc)*100;

            fprintf('\n=== Overall Accuracy for Protocol %s, CSP Filter %d ===\n', protocolName, filt);
            fprintf('Mean Accuracy: %.2f%% ± %.2f%%\n', overallAcc, overallStd);
            
            % Add as row to the protocol results table
            overallRow = table( ...
                string(protocolName), ...
                filt, ...
                "Overall Accuracy", ...
                overallAcc, ...
                overallStd, ...
                'VariableNames', {'Protocol', 'CSPFilters', 'Comparison', 'MeanAccuracy', 'StdAccuracy'} ...
            );
            
            protocolResults = [protocolResults; overallRow];
        end

        % --- Save results for this protocol ---
        % Clean protocol name for filename (remove spaces and special characters)
        cleanProtocolName = regexprep(protocolName, '[^\w]', '_');
        outputFile = fullfile(resultsFolder, sprintf('%s_results.xlsx', cleanProtocolName));
        
        try
            writetable(protocolResults, outputFile);
            fprintf('\nResults saved to: %s\n', outputFile);
            disp(protocolResults);
        catch ME
            fprintf('Error saving results for protocol %s: %s\n', protocolName, ME.message);
        end
    else
        fprintf('Warning: No results generated for protocol %s\n', protocolName);
    end
end

fprintf('\n=== Processing Complete ===\n');
fprintf('All protocol results have been saved to: %s\n', resultsFolder);

% === Collect and display overall accuracy summary ===
overallSummary = [];
currentRunProtocols = {}; % Track protocols processed in current run

% Store protocols processed in current run
for protocolIdx = 1:length(protocolFolders)
    currentRunProtocols{end+1} = protocolFolders(protocolIdx).name; %#ok<SAGROW>
end

% Re-scan all generated result files to collect overall accuracy data
resultFiles = dir(fullfile(resultsFolder, '*_results.xlsx'));

if ~isempty(resultFiles)
    fprintf('\n=== OVERALL ACCURACY SUMMARY (Current Run Only) ===\n');
    
    for fileIdx = 1:length(resultFiles)
        try
            % Read the results file
            filePath = fullfile(resultFiles(fileIdx).folder, resultFiles(fileIdx).name);
            data = readtable(filePath);
            
            % Extract protocol name from filename
            [~, fileName, ~] = fileparts(resultFiles(fileIdx).name);
            protocolName = strrep(fileName, '_results', '');
            
            % Find overall accuracy rows
            overallIdx = strcmp(data.Comparison, "Overall Accuracy");
            
            if any(overallIdx)
                overallData = data(overallIdx, :);
                
                % Only display if this protocol was processed in current run
                if ismember(protocolName, currentRunProtocols)
                    % Display results for this protocol
                    fprintf('\n--- Protocol: %s ---\n', protocolName);
                    for i = 1:height(overallData)
                        fprintf('  CSP Filter %d: %.2f%% ± %.2f%%\n', ...
                            overallData.CSPFilters(i), ...
                            overallData.MeanAccuracy(i), ...
                            overallData.StdAccuracy(i));
                    end
                end
                
                % Add to summary table (keep all protocols for the table)
                for i = 1:height(overallData)
                    summaryRow = table( ...
                        string(protocolName), ...
                        overallData.CSPFilters(i), ...
                        overallData.MeanAccuracy(i), ...
                        overallData.StdAccuracy(i), ...
                        'VariableNames', {'Protocol', 'CSPFilter', 'OverallAccuracy', 'StdDeviation'} ...
                    );
                    overallSummary = [overallSummary; summaryRow];
                end
            end
            
        catch ME
            fprintf('Error reading results file %s: %s\n', resultFiles(fileIdx).name, ME.message);
        end
    end
    
    % Save overall summary table
    if ~isempty(overallSummary)
        summaryFile = fullfile(resultsFolder, 'Overall_Accuracy_Summary.xlsx');
        try
            writetable(overallSummary, summaryFile);
            fprintf('\n=== SUMMARY TABLE ===\n');
            disp(overallSummary);
            fprintf('\nOverall summary saved to: %s\n', summaryFile);
            
            % Display formatted summary by filter
            fprintf('\n=== SUMMARY BY CSP FILTER ===\n');
            uniqueFilters = unique(overallSummary.CSPFilter);
            for filt = uniqueFilters'
                fprintf('\n--- CSP Filter %d ---\n', filt);
                filterIdx = overallSummary.CSPFilter == filt;
                filterData = overallSummary(filterIdx, :);
                
                for i = 1:height(filterData)
                    fprintf('  %s: %.2f%% ± %.2f%%\n', ...
                        filterData.Protocol(i), ...
                        filterData.OverallAccuracy(i), ...
                        filterData.StdDeviation(i));
                end
                
                % Calculate mean across protocols for this filter
                meanAcc = mean(filterData.OverallAccuracy);
                meanStd = mean(filterData.StdDeviation);
                fprintf('  --> Average across protocols: %.2f%% ± %.2f%%\n', meanAcc, meanStd);
            end
            
        catch ME
            fprintf('Error saving summary table: %s\n', ME.message);
            fprintf('\n=== SUMMARY TABLE ===\n');
            disp(overallSummary);
        end
    else
        fprintf('Warning: No overall accuracy data found to summarize.\n');
    end
else
    fprintf('Warning: No result files found to summarize.\n');
end