folderName = 'C:\Users\LENOVO\Documents\Uni\Master Thesis\22_05_ISR_Data\Datasets\Marko_25_Rest_first_HOH';
csp_para = 2;            % number of CSP components per class
fs = 256;                % sampling frequency (Hz)
start_sec = 0.5;         % start time for segment (seconds)
end_sec = 2.5;           % end time for segment (seconds)
class1 = 1;              % class label 1
class2 = 3;              % class label 2
N_channels = 16;         % number of EEG channels
Window_buff = 256;       % buffer size (probably equals fs)
T_task_class = 6;        % duration of task per class (seconds)
classes = [1, 3, 4];     % all classes in data
chunk_length_sec = 6;    % chunk length in seconds

[cspw, model] = ac_classification_model(folderName, csp_para, fs, start_sec, end_sec, class1, class2, ...
                                        N_channels, Window_buff, T_task_class, classes, chunk_length_sec);
