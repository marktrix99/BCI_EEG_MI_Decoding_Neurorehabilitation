function [X_all, y_all] = load_eeg_data(folderName, N_channels, Window_buff, T_task_class, classes, chunk_length_sec, fs)
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