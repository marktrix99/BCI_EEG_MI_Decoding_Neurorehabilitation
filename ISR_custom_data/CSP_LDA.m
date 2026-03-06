function [acc,left_num,right_num] = CSP_LDA(MIEEGData,label,Fs,LowFreq,UpFreq)
%  Common Spatial Pattern + Linear Discriminant Analysis
%  Input:
%  MIEEGData:The raw motor imagery data
%  label: motor imagery data label, 1 and 2
%  Fs: the sampling frequency (in Hz)
%  LowFreq: High pass filtering parameter settings
%  UpFreq:Low pass filtering parameter settings
%  Output:
%  acc:Accuracy Rate
%  left_num:
%  right_num:

% Obtaining effective channel EEG data
channel = [1:17 19:30];
trigger = find(MIEEGData(:,33)==2);
eeg = zeros(2000,29,40);
% Divide data into three dimensions based on Trigger
for i = 1:length(trigger)
    R = MIEEGData(trigger(i)-800:trigger(i)-800+2799,channel)';
    NotchData = NotchFilter(R,Fs,50);   % Notch filtering out 50Hz power frequency
    BandpassData = BandpassFilter(NotchData,Fs,LowFreq,UpFreq)';   % Band Pass Filter
    eeg(:,:,i) = BandpassData(801:2800,:);
end

% Parameters to be returned
acc = [];
left_num = [];
right_num = [];

% Cross validation partitioning dataset
for h = 1:10
    % Partition Dataset
    num = 1:40;
    num = num(randperm(length(num)));   % Disrupting 40 sets of data
    train_num = num(1:24);
    test_num = num(25:40);
    eeg_train = eeg(:,:,train_num);
    eeg_test = eeg(:,:,test_num);
    label_train = label(train_num);
    label_test = label(test_num);
    
    % CSPfeature
    csp_para = 2;       % CSP feature: 2*csp_para
    [train_feature, cspw] = CSP(eeg_train, label_train, csp_para);
    
    % Extract features from the test set
    test_feature = CSPFilter(eeg_test,cspw,csp_para);
    
    % LDA
    model = fitcdiscr(train_feature,label_train);
    test_label = predict(model,test_feature(:,1:4));

    % Calculate accuracy
    c = 0;
    L = 0;
    R = 0;
    for k = 1:length(label_test)
        if test_label(k)==label_test(k)
            c = c + 1;
            if test_label(k)==1
                L = L + 1;
            end
            if test_label(k)==2
                R = R +1;
            end
        end
        left = length(find(label_test==1));
        right = length(find(label_test==2));
    end
    acc = [acc,(c)/length(label_test)*100];
    left_num = [left_num;left,L];
    right_num = [right_num;right,R];
end
end