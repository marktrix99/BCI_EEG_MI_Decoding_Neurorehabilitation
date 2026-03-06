function [X_tr, X_te] = split_trials(X, pct)
    n = size(X,1);
    idx = randperm(n);
    n_tr = round(pct * n);
    X_tr = X(idx(1:n_tr),:,:);
    X_te = X(idx(n_tr+1:end),:,:);
end

function [W, P] = compute_csp(X1, X2)
    C1 = cov_avg(X1);
    C2 = cov_avg(X2);
    Cc = C1 + C2;
    [V, D] = eig(Cc);
    P = sqrtm(inv(D)) * V';
    S1 = P * C1 * P';
    [B, ~] = eig(S1);
    W = B' * P;
end

function C = cov_avg(X)
    [n, ch, ~] = size(X);
    C = zeros(ch);
    for i = 1:n
        trial = squeeze(X(i,:,:));
        c = (trial * trial') / trace(trial * trial');
        C = C + c;
    end
    C = C / n;
end

function Z = project_trials(X, W, m)
    Wm = [W(1:m,:); W(end-m+1:end,:)];
    n = size(X,1);
    Z = zeros(n, 2*m, size(X,3));
    for i = 1:n
        Z(i,:,:) = Wm * squeeze(X(i,:,:));
    end
end

function feat = extract_features(Z)
    n = size(Z,1);
    feat = zeros(n, size(Z,2));
    for i = 1:n
        varZ = var(squeeze(Z(i,:,:))');
        feat(i,:) = log10(varZ / sum(varZ));
    end
end

