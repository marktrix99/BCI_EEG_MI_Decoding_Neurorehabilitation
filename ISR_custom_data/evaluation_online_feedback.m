%%% Online feedback Evaluation: Total, per trial, per interval %%%

% Parameters
fs = 256; % Sampling frequency (Hz)
window_duration = 1.5; % Duration of each analysis window (s)
step_duration = 0.5; % Step size (s)
start_point = 0.5;               
window_length = round(window_duration * fs);
step_size = round(step_duration * fs);

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

% Initialize accuracy tracking
num_trials = length(valid_starts);
accuracy_per_trial = zeros(1, num_trials);
accuracy_at_least_one = zeros(1, num_trials); % one at least
accuracy_first_window = zeros(1, num_trials); % first window
correct_count = 0;
total_windows = 0;

% Plotting trials
figure('Position', [100 100 1200 600]);
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
    
    accuracy_per_trial(i) = mean(matches);
    accuracy_at_least_one(i) = any(matches); % 1 if any window correct
    accuracy_first_window(i) = matches(1); % check first window only
    
    % ==== Plot ====
    x_vals = 0:(trial_end - trial_start);
    subplot(ceil(num_trials/4), 4, i);
    plot(x_vals/fs, y(18, trial_start:trial_end), 'b', 'LineWidth', 1.5); hold on;
    plot(x_vals/fs, y(19, trial_start:trial_end), 'r', 'LineWidth', 1.5);
    
    % Highlight each window - simplified version
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
        
        % Draw simple dashed lines for window boundaries
        xline(x1, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5); % gray dashed for start
        xline(x2, '--', 'Color', line_color, 'LineWidth', 1.5); % colored solid for end
    end
    
    % Add annotation if at least one window was correct
    if accuracy_at_least_one(i)
        text(0.5, 4.2, '✓', 'Color', 'g', 'FontSize', 20, 'HorizontalAlignment', 'center');
    else
        text(0.5, 4.2, '✗', 'Color', 'r', 'FontSize', 20, 'HorizontalAlignment', 'center');
    end
    
    title(sprintf('Trial %d: %.0f%% (Any: %d)', i, 100*accuracy_per_trial(i), accuracy_at_least_one(i)));
    ylim([-0.5 4.5]);
    xlabel('Time (s)');
    grid on;
    if i == 1
        legend('True (18)', 'Measured (19)');
    end
    % ==== Print per-trial result ====
fprintf('Trial %d: %d correct out of %d windows (%.1f%%) | At least one: %d | First window: %d\n' , ...
    i, sum(matches), num_steps, 100*accuracy_per_trial(i), accuracy_at_least_one(i), accuracy_first_window(i));
end

% Overall Accuracy calculations
overall_accuracy = correct_count / total_windows * 100;
overall_at_least_one = mean(accuracy_at_least_one) * 100; 
overall_first_window = mean(accuracy_first_window) * 100;  

% Accuracy Plot
figure;
subplot(3,1,1); % Changed to 3 rows
plot(accuracy_per_trial, '-o', 'LineWidth', 2, 'MarkerSize', 8);
ylim([0 1.2]);
yticks([0 1]);
yticklabels({'Incorrect', 'Correct'});
xlabel('Trial Number');
ylabel('Avg. Accuracy');
title(sprintf('Window-by-Window Accuracy (Overall: %.1f%%)', overall_accuracy));
grid on;

subplot(3,1,2);
stem(accuracy_at_least_one, 'LineWidth', 2, 'MarkerSize', 8);
ylim([-0.2 1.2]);
yticks([0 1]);
yticklabels({'No', 'Yes'});
xlabel('Trial Number');
ylabel('Any Correct');
title(sprintf('Any Correct Window (Overall: %.1f%%)', overall_at_least_one));
grid on;

subplot(3,1,3); % New subplot for first window accuracy
stem(accuracy_first_window, 'LineWidth', 2, 'MarkerSize', 8);
ylim([-0.2 1.2]);
yticks([0 1]);
yticklabels({'No', 'Yes'});
xlabel('Trial Number');
ylabel('First Correct');
title(sprintf('First Window Correct (Overall: %.1f%%)', overall_first_window));
grid on;


% Class distribution
fprintf('\nAnalysis Results (Sliding Window Evaluation):\n');
fprintf('Trials analyzed: %d\n', num_trials);
fprintf('Total windows analyzed: %d\n', total_windows);
fprintf('Correct windows: %d\n', correct_count);
fprintf('Overall window accuracy: %.2f%%\n', overall_accuracy);
fprintf('Trials with at least one correct window: %.2f%%\n', overall_at_least_one);
fprintf('Trials with correct first window: %.2f%%\n', overall_first_window); % New line
fprintf('Window size: %.1fs, Step size: %.1fs\n', window_duration, step_duration);
fprintf('\nClass distribution:\n');