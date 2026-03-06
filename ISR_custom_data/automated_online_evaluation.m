%%% Automated Online Feedback Evaluation for All Protocols %%%

% === Configuration ===
fs = 256; % Sampling frequency (Hz)
window_duration = 1.5; % Duration of each analysis window (s)
step_duration = 0.5; % Step size (s)
start_point = 0.5;               
window_length = round(window_duration * fs);
step_size = round(step_duration * fs);

% === Directory Setup ===
baseDirectory = 'C:\Users\LENOVO\Documents\Uni\Master Thesis\ISR_Data';
testDate = 'Test_31_07_2025'; % Data loading
onlineFeedbackPath = fullfile(baseDirectory, testDate, 'Teste Online');
resultsFolder = fullfile(baseDirectory, 'Overall_Results_Online_Feedback');

% Create results folder if it doesn't exist
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% === Find all participant folders ===
participantFolders = dir(onlineFeedbackPath);
participantFolders = participantFolders([participantFolders.isdir]); % only directories
participantFolders = participantFolders(~ismember({participantFolders.name}, {'.', '..'})); % exclude . and ..

if isempty(participantFolders)
    error('No participant folders found in: %s', onlineFeedbackPath);
end

fprintf('Found %d participant folders for online feedback:\n', length(participantFolders));
for i = 1:length(participantFolders)
    fprintf('  %d: %s\n', i, participantFolders(i).name);
end

% === Initialize results table ===
onlineResults = [];

% === Process each participant ===
for participantIdx = 1:length(participantFolders)
    participantName = participantFolders(participantIdx).name;
    participantPath = fullfile(onlineFeedbackPath, participantName);
    
    % Extract participant base name and determine protocol type
    if contains(participantName, '_HOH')
        protocolType = 'HOH';
        participantBase = strrep(participantName, '_HOH', '');
    elseif contains(participantName, '_real_movement')
        protocolType = 'real_movement';
        participantBase = strrep(participantName, '_real_movement', '');
    else
        protocolType = 'overview';
        participantBase = participantName;
    end
    
    fprintf('\n=== Processing: %s (Protocol: %s) ===\n', participantName, protocolType);
    
    % Find the random-numbered folder inside participant folder
    subFolders = dir(participantPath);
    subFolders = subFolders([subFolders.isdir]); % only directories
    subFolders = subFolders(~ismember({subFolders.name}, {'.', '..'})); % exclude . and ..
    
    if isempty(subFolders)
        fprintf('Warning: No subfolders found in %s\n', participantPath);
        continue;
    end
    
    % Look for signals.mat in each subfolder (usually there's only one random-numbered folder)
    signalsFound = false;
    for subIdx = 1:length(subFolders)
        subFolderPath = fullfile(participantPath, subFolders(subIdx).name);
        matFiles = dir(fullfile(subFolderPath, '*signals.mat'));
        
        if ~isempty(matFiles)
            signalsFound = true;
            % Use the first signals.mat file found
            dataFile = fullfile(subFolderPath, matFiles(1).name);
            fprintf('Found signals.mat in subfolder: %s\n', subFolders(subIdx).name);
            fprintf('Loading: %s\n', dataFile);
            break;
        end
    end
    
    if ~signalsFound
        fprintf('Warning: No signals.mat file found in any subfolder of %s\n', participantPath);
        continue;
    end
    
    try
        load(dataFile, 'y');
        
        if ~exist('y', 'var') || size(y, 1) < 19
            fprintf('Error: Invalid data format in %s (need at least 19 channels)\n', dataFile);
            continue;
        end
        
        % === Analysis (same as original code) ===
        % Initialize
        valid_starts = [];
        valid_labels = [];
        
        state = 0;  % 0 = waiting for movement start, 1 = in movement
        signal_len = size(y, 2);
        
        % Find event-based trials: starts at 3 or 4, ends at 2
        for i = 2:signal_len
            if state == 0 && (y(18,i) == 3 || y(18,i) == 4)
                trial_start_idx = i;
                current_label = y(18,i);
                state = 1;
            elseif state == 1 && y(18,i) == 2
                trial_end_idx = i - 1;
                if trial_end_idx > trial_start_idx
                    valid_starts(end+1) = trial_start_idx;
                    valid_labels(end+1) = current_label;
                end
                state = 0;
            end
        end
        
        if isempty(valid_starts)
            fprintf('Warning: No valid trials found in %s\n', dataFile);
            continue;
        end
        
        % Initialize accuracy tracking
        num_trials = length(valid_starts);
        accuracy_per_trial = zeros(1, num_trials);
        accuracy_at_least_one = zeros(1, num_trials);
        accuracy_first_window = zeros(1, num_trials);
        correct_count = 0;
        total_windows = 0;
        
        % === Analysis per trial ===
        for i = 1:num_trials
            trial_start = valid_starts(i);
            true_label = valid_labels(i);
            
            % Trial end: find when label changes from current true_label
            trial_end = trial_start;
            while trial_end <= signal_len && y(18, trial_end) == true_label
                trial_end = trial_end + 1;
            end
            trial_end = trial_end - 1;
            
            trial_length = trial_end - trial_start + 1;
            num_steps = floor((trial_length - window_length) / step_size) + 1;
            if num_steps <= 0
                continue;  % Skip short trials
            end
            
            matches = zeros(1, num_steps);
            win_starts = zeros(1, num_steps);
            
            for s = 0:(num_steps-1)
                offset = s * step_size + round(start_point * fs);
                win_start = trial_start + offset;
                win_end = win_start + window_length - 1;
                
                % Ensure window is within bounds
                if win_end > trial_end
                    num_steps = s;  % Adjust total number of steps
                    matches = matches(1:num_steps);
                    win_starts = win_starts(1:num_steps);
                    break;
                end
                
                % Check only the LAST sample of the window
                measured_value = y(19, win_end);
                true_value_at_end = y(18, win_end);
                
                is_match = (measured_value == true_value_at_end);
                matches(s+1) = is_match;
                win_starts(s+1) = win_start;
                correct_count = correct_count + is_match;
                total_windows = total_windows + 1;
            end
            
            if num_steps > 0
                accuracy_per_trial(i) = mean(matches);
                accuracy_at_least_one(i) = any(matches);
                accuracy_first_window(i) = matches(1);
            end
        end
        
        % Calculate overall metrics
        overall_accuracy = correct_count / total_windows * 100;
        overall_at_least_one = mean(accuracy_at_least_one) * 100;
        overall_first_window = mean(accuracy_first_window) * 100;
        
        % Display results
        fprintf('\nResults for %s (%s protocol):\n', participantBase, protocolType);
        fprintf('Trials analyzed: %d\n', num_trials);
        fprintf('Total windows analyzed: %d\n', total_windows);
        fprintf('Overall window accuracy: %.2f%%\n', overall_accuracy);
        fprintf('Trials with at least one correct window: %.2f%%\n', overall_at_least_one);
        fprintf('Trials with correct first window: %.2f%%\n', overall_first_window);
        
        % === Create visualization for this participant ===
        figure('Position', [100 100 1200 800], 'Name', sprintf('%s - %s Protocol', participantBase, protocolType));
        
        % Calculate subplot arrangement
        numTrialsToShow = min(num_trials, 20); % Show up to 20 trials
        cols = 4;
        rows = ceil(numTrialsToShow / cols);
        
        % Plot trials
        for i = 1:numTrialsToShow
            trial_start = valid_starts(i);
            true_label = valid_labels(i);
            
            % Find trial end
            trial_end = trial_start;
            while trial_end <= signal_len && y(18, trial_end) == true_label
                trial_end = trial_end + 1;
            end
            trial_end = trial_end - 1;
            
            trial_length = trial_end - trial_start + 1;
            num_steps = floor((trial_length - window_length) / step_size) + 1;
            if num_steps <= 0
                continue;  % Skip short trials
            end
            
            % Recalculate matches for plotting (same as analysis above)
            matches = zeros(1, num_steps);
            win_starts = zeros(1, num_steps);
            
            for s = 0:(num_steps-1)
                offset = s * step_size + round(start_point * fs);
                win_start = trial_start + offset;
                win_end = win_start + window_length - 1;
                
                % Ensure window is within bounds
                if win_end > trial_end
                    num_steps = s;  % Adjust total number of steps
                    matches = matches(1:num_steps);
                    win_starts = win_starts(1:num_steps);
                    break;
                end
                
                % Check only the LAST sample of the window
                measured_value = y(19, win_end);
                true_value_at_end = y(18, win_end);
                
                is_match = (measured_value == true_value_at_end);
                matches(s+1) = is_match;
                win_starts(s+1) = win_start;
            end
            
            % Plot this trial
            subplot(rows, cols, i);
            x_vals = 0:(trial_end - trial_start);
            plot(x_vals/fs, y(18, trial_start:trial_end), 'b', 'LineWidth', 1.5); hold on;
            plot(x_vals/fs, y(19, trial_start:trial_end), 'r', 'LineWidth', 1.5);
            
            % Highlight each window
            for s = 1:num_steps
                if win_starts(s) == 0, continue; end
                x1 = (win_starts(s) - trial_start)/fs;
                x2 = x1 + window_duration;
                
                % Determine color based on match at END of window
                if matches(s) == 1
                    line_color = 'g'; % green for correct
                else
                    line_color = 'r'; % red for incorrect
                end
                
                % Draw window boundaries
                xline(x1, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5); % gray dashed for start
                xline(x2, '--', 'Color', line_color, 'LineWidth', 1.5); % colored for end
            end
            
            % Add checkmark/X annotation
            if accuracy_at_least_one(i)
                text(0.1, 4.2, '✓', 'Color', 'g', 'FontSize', 16, 'FontWeight', 'bold');
            else
                text(0.1, 4.2, '✗', 'Color', 'r', 'FontSize', 16, 'FontWeight', 'bold');
            end
            
            title(sprintf('Trial %d: %.0f%% (Any: %d)', i, 100*accuracy_per_trial(i), accuracy_at_least_one(i)));
            ylim([-0.5 4.5]);
            xlabel('Time (s)');
            ylabel('Label');
            grid on;
            
            if i == 1
                legend('True (18)', 'Measured (19)', 'Location', 'best');
            end
        end
        
        % === Add to results table ===
        participantRow = table( ...
            string(participantBase), ...
            string(protocolType), ...
            num_trials, ...
            total_windows, ...
            correct_count, ...
            overall_accuracy, ...
            overall_at_least_one, ...
            overall_first_window, ...
            'VariableNames', {'Participant', 'Protocol', 'NumTrials', 'TotalWindows', 'CorrectWindows', ...
                            'OverallWindowAccuracy', 'AtLeastOneCorrect', 'FirstWindowCorrect'} ...
        );
        
        onlineResults = [onlineResults; participantRow];
        
    catch ME
        fprintf('Error processing %s: %s\n', dataFile, ME.message);
        continue;
    end
end

% === Save and display final results ===
if ~isempty(onlineResults)
    % Add test date to results for consolidation
    onlineResults.TestDate = repmat(string(testDate), height(onlineResults), 1);
    
    % Define consolidated results file (across all test dates)
    consolidatedResultsFile = fullfile(resultsFolder, 'Consolidated_Online_Feedback_Results.xlsx');
    
    % Load existing consolidated results if they exist
    consolidatedResults = [];
    if exist(consolidatedResultsFile, 'file')
        try
            consolidatedResults = readtable(consolidatedResultsFile);
            fprintf('Loading existing consolidated results...\n');
            
            % Remove entries from current test date to avoid duplicates
            consolidatedResults = consolidatedResults(~strcmp(consolidatedResults.TestDate, testDate), :);
        catch ME
            fprintf('Warning: Could not load existing consolidated results: %s\n', ME.message);
            consolidatedResults = [];
        end
    end
    
    % Combine with current results
    if isempty(consolidatedResults)
        consolidatedResults = onlineResults;
    else
        % Ensure column order matches
        if ~isequal(consolidatedResults.Properties.VariableNames, onlineResults.Properties.VariableNames)
            % Reorder columns to match
            onlineResults = onlineResults(:, consolidatedResults.Properties.VariableNames);
        end
        consolidatedResults = [consolidatedResults; onlineResults];
    end
    
    % Save individual test date results
    resultsFile = fullfile(resultsFolder, sprintf('Online_Feedback_Results_%s.xlsx', testDate));
    try
        writetable(onlineResults, resultsFile);
        fprintf('\n=== CURRENT TEST DATE RESULTS SAVED ===\n');
        fprintf('Results saved to: %s\n', resultsFile);
    catch ME
        fprintf('Error saving individual results: %s\n', ME.message);
    end
    
    % Save consolidated results
    try
        writetable(consolidatedResults, consolidatedResultsFile);
        fprintf('\n=== CONSOLIDATED RESULTS SAVED ===\n');
        fprintf('Consolidated results saved to: %s\n', consolidatedResultsFile);
        fprintf('Total entries in consolidated table: %d\n', height(consolidatedResults));
    catch ME
        fprintf('Error saving consolidated results: %s\n', ME.message);
    end
    
    % Display current test date results
    fprintf('\n=== CURRENT TEST DATE RESULTS (%s) ===\n', testDate);
    disp(onlineResults);
    
    % Display consolidated results summary
    fprintf('\n=== CONSOLIDATED RESULTS SUMMARY ===\n');
    disp(consolidatedResults);
    
    % Protocol-based summary statistics for current test date
    fprintf('\n=== PROTOCOL-BASED SUMMARY (%s) ===\n', testDate);
    fprintf('Number of datasets: %d\n', height(onlineResults));
    
    % Get unique protocols for current test date
    uniqueProtocols = unique(onlineResults.Protocol);
    
    for i = 1:length(uniqueProtocols)
        protocol = uniqueProtocols(i);
        protocolIdx = strcmp(onlineResults.Protocol, protocol);
        protocolData = onlineResults(protocolIdx, :);
        
        fprintf('\n--- %s Protocol (%s) ---\n', protocol, testDate);
        fprintf('Number of participants: %d\n', sum(protocolIdx));
        fprintf('Average overall window accuracy: %.2f%% ± %.2f%%\n', ...
            mean(protocolData.OverallWindowAccuracy), std(protocolData.OverallWindowAccuracy));
        fprintf('Average at least one correct: %.2f%% ± %.2f%%\n', ...
            mean(protocolData.AtLeastOneCorrect), std(protocolData.AtLeastOneCorrect));
        fprintf('Average first window correct: %.2f%% ± %.2f%%\n', ...
            mean(protocolData.FirstWindowCorrect), std(protocolData.FirstWindowCorrect));
    end
    
    % Cross-test-date analysis if we have multiple test dates
    allTestDates = unique(consolidatedResults.TestDate);
    if length(allTestDates) > 1
        fprintf('\n=== CROSS-TEST-DATE ANALYSIS ===\n');
        
        for dateIdx = 1:length(allTestDates)
            currentDate = allTestDates(dateIdx);
            dateData = consolidatedResults(strcmp(consolidatedResults.TestDate, currentDate), :);
            
            fprintf('\n--- Test Date: %s ---\n', currentDate);
            fprintf('Total participants: %d\n', height(dateData));
            fprintf('Average overall window accuracy: %.2f%% ± %.2f%%\n', ...
                mean(dateData.OverallWindowAccuracy), std(dateData.OverallWindowAccuracy));
            fprintf('Average at least one correct: %.2f%% ± %.2f%%\n', ...
                mean(dateData.AtLeastOneCorrect), std(dateData.AtLeastOneCorrect));
            fprintf('Average first window correct: %.2f%% ± %.2f%%\n', ...
                mean(dateData.FirstWindowCorrect), std(dateData.FirstWindowCorrect));
        end
        
        % Overall consolidated summary
        fprintf('\n--- OVERALL CONSOLIDATED SUMMARY (All Test Dates) ---\n');
        fprintf('Total participants across all dates: %d\n', height(consolidatedResults));
        fprintf('Average overall window accuracy: %.2f%% ± %.2f%%\n', ...
            mean(consolidatedResults.OverallWindowAccuracy), std(consolidatedResults.OverallWindowAccuracy));
        fprintf('Average at least one correct: %.2f%% ± %.2f%%\n', ...
            mean(consolidatedResults.AtLeastOneCorrect), std(consolidatedResults.AtLeastOneCorrect));
        fprintf('Average first window correct: %.2f%% ± %.2f%%\n', ...
            mean(consolidatedResults.FirstWindowCorrect), std(consolidatedResults.FirstWindowCorrect));
    else
        % Current test date summary
        fprintf('\n--- Overall Summary (%s) ---\n', testDate);
        fprintf('Average overall window accuracy: %.2f%% ± %.2f%%\n', ...
            mean(onlineResults.OverallWindowAccuracy), std(onlineResults.OverallWindowAccuracy));
        fprintf('Average at least one correct: %.2f%% ± %.2f%%\n', ...
            mean(onlineResults.AtLeastOneCorrect), std(onlineResults.AtLeastOneCorrect));
        fprintf('Average first window correct: %.2f%% ± %.2f%%\n', ...
            mean(onlineResults.FirstWindowCorrect), std(onlineResults.FirstWindowCorrect));
    end
    
    % Create visualization for current test date results
    figure('Position', [300 300 1000 450], 'Name', sprintf('Protocol Comparison - %s', testDate));
    
    % Create grouped bar chart for current test date
    protocols = categorical(onlineResults.Protocol);
    participants = onlineResults.Participant;
    
    % Combine participant and protocol for x-axis labels
    xLabels = strcat(participants, ' (', string(protocols), ')');
    
    subplot(1,3,1);
    b1 = bar(onlineResults.OverallWindowAccuracy);
    set(gca, 'XTickLabel', xLabels, 'XTickLabelRotation', 45);
    ylabel('Accuracy (%)');
    title(sprintf('Overall Window Accuracy - %s', testDate));
    grid on;
    
    % Color bars by protocol
    hold on;
    for i = 1:length(uniqueProtocols)
        protocolIdx = strcmp(onlineResults.Protocol, uniqueProtocols(i));
        if strcmp(uniqueProtocols(i), 'overview')
            barColor = [0.2 0.6 0.8]; % Blue
        elseif strcmp(uniqueProtocols(i), 'HOH')
            barColor = [0.8 0.4 0.2]; % Orange
        else % real_movement
            barColor = [0.2 0.8 0.4]; % Green
        end
        b1.CData(protocolIdx, :) = repmat(barColor, sum(protocolIdx), 1);
    end
    
    subplot(1,3,2);
    b2 = bar(onlineResults.AtLeastOneCorrect);
    set(gca, 'XTickLabel', xLabels, 'XTickLabelRotation', 45);
    ylabel('Accuracy (%)');
    title(sprintf('At Least One Correct - %s', testDate));
    grid on;
    
    % Color bars by protocol
    hold on;
    for i = 1:length(uniqueProtocols)
        protocolIdx = strcmp(onlineResults.Protocol, uniqueProtocols(i));
        if strcmp(uniqueProtocols(i), 'overview')
            barColor = [0.2 0.6 0.8]; % Blue
        elseif strcmp(uniqueProtocols(i), 'HOH')
            barColor = [0.8 0.4 0.2]; % Orange
        else % real_movement
            barColor = [0.2 0.8 0.4]; % Green
        end
        b2.CData(protocolIdx, :) = repmat(barColor, sum(protocolIdx), 1);
    end
    
    subplot(1,3,3);
    b3 = bar(onlineResults.FirstWindowCorrect);
    set(gca, 'XTickLabel', xLabels, 'XTickLabelRotation', 45);
    ylabel('Accuracy (%)');
    title(sprintf('First Window Correct - %s', testDate));
    grid on;
    
    % Color bars by protocol
    hold on;
    for i = 1:length(uniqueProtocols)
        protocolIdx = strcmp(onlineResults.Protocol, uniqueProtocols(i));
        if strcmp(uniqueProtocols(i), 'overview')
            barColor = [0.2 0.6 0.8]; % Blue
        elseif strcmp(uniqueProtocols(i), 'HOH')
            barColor = [0.8 0.4 0.2]; % Orange
        else % real_movement
            barColor = [0.2 0.8 0.4]; % Green
        end
        b3.CData(protocolIdx, :) = repmat(barColor, sum(protocolIdx), 1);
    end
    
    % Add legend
    if length(uniqueProtocols) > 1
        legend(cellstr(uniqueProtocols), 'Location', 'best');
    end
    
else
    fprintf('Warning: No results generated.\n');
end

fprintf('\n=== ONLINE FEEDBACK ANALYSIS COMPLETE ===\n');