 function [X1, X2, class2_label] = separate_classes(X_all, y_all, class1, class2)
        % First ensure we have valid indices
        class1_idx = find(y_all == class1);
        if isempty(class1_idx)
            error('No trials found for class %d', class1);
        end
        
        if isscalar(class2)
            class2_idx = find(y_all == class2);
            class2_label = sprintf('Class %d', class2);
        else
            class2_idx = find(ismember(y_all, class2));
            class2_label = sprintf('Classes [%s]', num2str(class2));
        end
        
        if isempty(class2_idx)
            error('No trials found for class(es) %s', class2_label);
        end
        
        X1 = X_all(class1_idx, :, :);
        X2 = X_all(class2_idx, :, :);
        
        disp(['Separated classes: ' num2str(class1) ' (n=' num2str(size(X1,1)) ...
              ') vs ' class2_label ' (n=' num2str(size(X2,1)) ')']);
    end