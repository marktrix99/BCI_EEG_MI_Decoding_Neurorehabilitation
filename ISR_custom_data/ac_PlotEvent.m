function ac_PlotEvent(data,sampleRate,smoothPara, ch1, ch2)
%input datar1:3D event EEG data, data format:[sample points*channels* trials]
%     labelr1:datar1 label
%       lower:High pass filtering parameter settings
%      higher:Low pass filtering parameter settings
%  sampleRate:the sampling frequency (in Hz)
% butterOrder:Filter order
%  smoothPara:Smoothing coefficient
%
% output figure:ERD/ERS image

R_start = 1;  
R_end = 2;    
[~, ~, trial] = size(data); 

% step2: stacked and average
C3 = 0; 
C4 = 0;
data = permute(data,[1 3 2]);       
data = data.^2;             % square  
for i = 1:size(data,2)      % stacked
    C3 = C3 + data(:,i,ch1); 
    C4 = C4 + data(:,i,ch2); 
end
C3 = C3/trial;               % average   
C4 = C4/trial;

% data = mean(data, 3);
% data = data.^2;
% C3 = data(:,14);
% C4 = data(:,15);

% step3: smoothing
C3 = smooth(C3,smoothPara);           
C4 = smooth(C4,smoothPara);
C3mean = C3(sampleRate * R_start:sampleRate * R_end - 1);       % baseline data            
C4mean = C4(sampleRate * R_start:sampleRate * R_end - 1);
C3m = 0;
C4m = 0;
for i = 1:sampleRate
    C3m = C3m + C3mean(i);
    C4m = C4m + C4mean(i);
end
C3m = C3m/sampleRate;
C4m = C4m/sampleRate;

% step4: cal ERD/ERS = A - R / R * 100%     
C3 = ((C3 - C3m)/C3m) * 100;                             
C4 = ((C4 - C4m)/C4m) * 100;

C3 = ((C3 - C3m)/C3m) * 100;                             
C4 = ((C4 - C4m)/C4m) * 100;

% plot figure
plot(C3,'r--','LineWidth',1);
hold on
plot(C4,'b','LineWidth',1);

h = legend('C3','C4','Location','SouthWest','Orientation','horizontal');
legend('boxoff');
set(h,'fontsize',8)

% xlim(gca,[500 2500]);
% ss = [min(C3(500:2500)), max(C3(500:2500)),min(C4(500:2500)) max(C4(500:2500))];
% ylim(gca,[min(ss)-10 max(ss)]);
% xlabel('Time(s)','FontSize', 12);
% ylabel('Amplitude(��V)','FontSize', 12);
% 
% % axis([0 100 -20 200]);
% set(gca,'XTick',[0 500 1000 1500 2000 2500],'XTickLabel',...
%     {'-1','0','1','2','3','4'},'FontSize', 12);
end