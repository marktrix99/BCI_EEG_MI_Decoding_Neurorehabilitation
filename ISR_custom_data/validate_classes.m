    function [class1, class2, n1, n2] = validate_classes(y_all, class1, class2, min_trials)
        unique_classes = unique(y_all);
        fprintf('Available classes in data: %s\n', mat2str(unique_classes'));
        
        % Handle class1
        n1 = sum(y_all == class1);
        if n1 < min_trials
            error('Insufficient trials (%d) for class %d (minimum %d required)', n1, class1, min_trials);
        end
        
        % Handle class2 (single class or combination)
        if isscalar(class2)
            n2 = sum(y_all == class2);
            if n2 < min_trials
                error('Insufficient trials (%d) for class %d (minimum %d required)', n2, class2, min_trials);
            end
        else % Combination like [3 4]
            n2 = sum(ismember(y_all, class2));
            if n2 < min_trials
                error('Insufficient trials (%d) for classes [%s] (minimum %d required)', ...
                      n2, num2str(class2), min_trials);
            end
        end
    end