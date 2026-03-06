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