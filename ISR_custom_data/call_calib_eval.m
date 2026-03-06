%constants and call function
N_channels = 16;
Window_buff = 256;
T_task_class_calib = [6, 2, 6, 6, 90];      % durations in seconds for classes [rest, relax, close hand, open hand, calibration long rest] 
classes = [1, 2, 3, 4, 5];            % class labels
chunk_length_sec = 6;
folderCalib1 = 'C:\Users\LENOVO\Documents\Uni\Master Thesis\ISR_Data\Test_09_07_2025\Calibração\Rute_26\20250709_161223';
folderCalib2 = 'C:\Users\LENOVO\Documents\Uni\Master Thesis\ISR_Data\Test_09_07_2025\Calibração\Rute_26\20250709_161941';
folderEval = 'C:\Users\LENOVO\Documents\Uni\Master Thesis\ISR_Data\Test_09_07_2025\Teste Online\Rute_26\20250709_162809';
start_sec = 0.5;
end_sec = 2;
movement_comb = [3 4];
fs = 256;


% Define filter settings to test
filter_list = [2, 4];

% Preallocate results
results = [];

for i = 1:length(filter_list)
    csp_para = filter_list(i);

    % Run models for each comparison
    %[~, ~, acc_closed, std_closed]     = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, 1, 3, N_channels, Window_buff, T_task_class, classes, chunk_length_sec);
    %[~, ~, acc_opened, std_opened]     = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, 1, 4, N_channels, Window_buff, T_task_class, classes, chunk_length_sec);
    %[~, ~, acc_movement, std_movement] = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, 1, movement_comb, N_channels, Window_buff, T_task_class, classes, chunk_length_sec);

    [~, ~, acc_closed, std_closed] = classification__evaluation_model_fbcsp(folderCalib1, folderCalib2, folderEval, csp_para, fs, start_sec, end_sec, 1, 3, N_channels, Window_buff, T_task_class_calib, classes, chunk_length_sec);
    [~, ~, acc_opened, std_opened] = classification__evaluation_model_fbcsp(folderCalib1, folderCalib2, folderEval, csp_para, fs, start_sec, end_sec, 1, 4, N_channels, Window_buff, T_task_class_calib, classes, chunk_length_sec);
    [~, ~, acc_movement, std_movement] = classification__evaluation_model_fbcsp(folderCalib1, folderCalib2, folderEval, csp_para, fs, start_sec, end_sec, 1, movement_comb, N_channels, Window_buff, [6, 2, 6, 6], classes, chunk_length_sec);
    
    % Store results as a table
    tmp_table = table( ...
        repmat(csp_para, 3, 1), ...
        ["Rest vs Closed"; "Rest vs Opened"; "Rest vs Movement"], ...
        [acc_closed; acc_opened; acc_movement], ...
        [std_closed; std_opened; std_movement], ...
        'VariableNames', {'CSPFilters', 'Comparison', 'MeanAccuracy', 'StdAccuracy'} ...
    );

    % Append to results
    results = [results; tmp_table];
end

% Save the final combined results table
writetable(results, 'accuracy_rest_middle_2.xlsx');

% Optional: display in console
disp(results)