% Load .mat file
rawdata = load('C:\Users\LENOVO\Documents\Uni\Master Thesis\ISR_Data\Test_03_07_2025\Calibração\Marko_03_07_rest_first\20250703_103517\signals.mat');
y = rawdata.y;

% Constants
N_channels = 16;
Window_buff = 256;
T_task_class = [60, 2, 6, 6];      % durations in seconds for classes 1–4, change 6 and 60 for different protocols (rest)
classes = [1, 2, 3, 4];            % class labels
seg_size = T_task_class * Window_buff;  % total samples per class
chunk_length_sec = 6;
chunk_len = chunk_length_sec * Window_buff;  % 1536 samples

% Extract label row and find event onsets
label_row = y(N_channels + 2, :);
event_onsets = find([0 diff(label_row)] ~= 0);
event_values = label_row(event_onsets);

% Class-segment mapping
class_seg_map = containers.Map(classes, seg_size);

% Initialize output containers
X_cell = {};
y_label = [];

for i = 1:length(event_onsets)
    cls = event_values(i);
    if ismember(cls, classes)
        seg_len = class_seg_map(cls);
        onset = event_onsets(i);
        available_len = size(y, 2) - onset + 1;
        trial_data = y(2:N_channels+1, onset : min(onset + seg_len - 1, size(y, 2)));
        total_samples = size(trial_data, 2);

        % Split into multiple chunks if trial longer than 6s
        if total_samples > chunk_len
            num_chunks = floor(total_samples / chunk_len);
            for j = 1:num_chunks
                start_idx = (j - 1) * chunk_len + 1;
                end_idx = j * chunk_len;
                chunk = trial_data(:, start_idx:end_idx);
                X_cell{end+1} = chunk;
                y_label(end+1) = cls;
            end
        else
            % Keep short trial as is, pad if needed
            if total_samples < chunk_len
                pad_len = chunk_len - total_samples;
                trial_data = [trial_data, zeros(N_channels, pad_len)];
                fprintf('Padded class %d trial with %d zeros\n', cls, pad_len);
            end
            X_cell{end+1} = trial_data;
            y_label(end+1) = cls;
        end
    end
end

% Convert to 3D array
num_trials = length(X_cell);
X = zeros(N_channels, chunk_len, num_trials);
for i = 1:num_trials
    X(:, :, i) = X_cell{i};
end

% Convert y_label to column vector
y_label = y_label(:);

% Generate event_onsets as sample indices
event_onsets = (0:length(y_label)-1) * chunk_len + 1;
event_values = y_label;

% Save
save('03_07_rest_first_calibracao_1.npz', 'X', 'y_label', 'event_onsets', 'event_values');
