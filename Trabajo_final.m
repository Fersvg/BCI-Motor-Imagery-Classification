%% START & CLEANUP
clearvars; close all; clc;

%% LOAD DATA
filename = 'v01.mat';
EEG = load(filename);

%% DEFINITIONS
fs = EEG.EEG.srate;
signals = detrend(EEG.EEG.data(1:15,:)', 'constant');

bands = [8 12; 12 16; 16 20; 20 24; 24 30];
n_bands = size(bands,1);
ch_names = {'F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','P7','P3','Pz','P4','P8'};
num_events = numel(EEG.EEG.event);
first_trial = 121;

%% CHANNEL INDEXES
idx_C3 = find(strcmp(ch_names,'C3'));
idx_Cz = find(strcmp(ch_names,'Cz'));
idx_C4 = find(strcmp(ch_names,'C4'));

idx_P3 = find(strcmp(ch_names,'P3'));
idx_P4 = find(strcmp(ch_names,'P4'));
idx_F3 = find(strcmp(ch_names,'F3'));
idx_F4 = find(strcmp(ch_names,'F4'));

%% ERD PIPELINE 
[b,a] = butter(4,[8 30]/(fs/2),'bandpass');
data_ERD = filtfilt(b,a,signals);

bad_channels = detect_bad_channels(data_ERD, fs);
data_ERD = apply_car(data_ERD, bad_channels);

%% MU & BETA FILTERS FOR CSP
[b_mu,a_mu]     = butter(4,[8 13]/(fs/2),'bandpass');
[b_beta,a_beta] = butter(4,[14 30]/(fs/2),'bandpass');

data_mu   = filtfilt(b_mu,a_mu,signals);
data_beta = filtfilt(b_beta,a_beta,signals);

data_mu   = apply_car(data_mu, bad_channels);
data_beta = apply_car(data_beta, bad_channels);

%% MRCP PIPELINE

[b,a] = butter(2, [0.3 3]/(fs/2), 'bandpass');
data_MRCP = filtfilt(b,a,signals);
data_MRCP = apply_car(data_MRCP, bad_channels);

%% INIT
X_model1 = [];
X_model2 = [];
y = [];

std_C3 = [];
std_Cz = [];
std_C4 = [];

X_trials_3ch = [];
X_trials_7ch = [];
y_trials = [];

X_trials_3ch_mu = [];
X_trials_3ch_beta = [];

X_trials_7ch_mu = [];
X_trials_7ch_beta = [];

MRCP_left = [];
MRCP_right = [];
MRCP_rest = [];

rejected = 0;
trial_count = 0;
%% LOOP

for ev = first_trial*2 : 2 : num_events
    
    start_sample = EEG.EEG.event(ev).latency;
    stop_sample = start_sample + 2*fs - 1;
    
    % BASELINE (2s antes)
    baseline_start = start_sample - 2*fs;
    baseline_end   = start_sample - 1;

    if stop_sample > size(signals,1) || baseline_start < 1
        continue;
    end
    
    trial_ERD = data_ERD(start_sample:stop_sample,:);
    trial_MRCP = data_MRCP(start_sample:stop_sample,:);
    baseline_ERD = data_ERD(baseline_start:baseline_end,:);
    trial_mu   = data_mu(start_sample:stop_sample,:);
    trial_beta = data_beta(start_sample:stop_sample,:);
    baseline_mu   = data_mu(baseline_start:baseline_end,:);
    baseline_beta = data_beta(baseline_start:baseline_end,:);
    
    trial_ERD = detrend(trial_ERD,'constant');
    trial_MRCP = detrend(trial_MRCP,'constant');
    baseline_ERD = detrend(baseline_ERD,'constant');
    trial_mu = detrend(trial_mu,'constant');
    trial_beta = detrend(trial_beta,'constant');
    baseline_mu = detrend(baseline_mu,'constant');
    baseline_beta = detrend(baseline_beta,'constant');

    % Ventana relevante (0.5s–1.5s)
    win = round(0.5*fs):round(1.5*fs);
    trial_ERD = trial_ERD(win,:);
    trial_MRCP = trial_MRCP(win,:);
    trial_mu   = trial_mu(win,:);
    trial_beta = trial_beta(win,:);
  
    trial_vector = trial_ERD(:);
    
    % Trial rejection
    bad = 0;
    sigC3 = trial_ERD(:,idx_C3);
    sigCz = trial_ERD(:,idx_Cz);
    sigC4 = trial_ERD(:,idx_C4);

    std_C3(end+1) = std(sigC3);
    std_Cz(end+1) = std(sigCz);
    std_C4(end+1) = std(sigC4);

    if max(abs(sigC3)) > 5 || std(sigC3) > 1
        bad = bad + 1;
    end
    if max(abs(sigCz)) > 5 || std(sigCz) > 1
        bad = bad + 1;
    end
    if max(abs(sigC4)) > 5 || std(sigC4) > 1
        bad = bad + 1;
    end

    if bad >= 2  
        rejected = rejected + 1;
        continue;
    end

    %% -------- ERD FEATURES --------
     % MU
    mu_C3 = (bandpower(trial_ERD(:,idx_C3),fs,[8 13]) - ...
             bandpower(baseline_ERD(:,idx_C3),fs,[8 13])) / ...
             bandpower(baseline_ERD(:,idx_C3),fs,[8 13]);

    mu_Cz = (bandpower(trial_ERD(:,idx_Cz),fs,[8 13]) - ...
             bandpower(baseline_ERD(:,idx_Cz),fs,[8 13])) / ...
             bandpower(baseline_ERD(:,idx_Cz),fs,[8 13]);

    mu_C4 = (bandpower(trial_ERD(:,idx_C4),fs,[8 13]) - ...
             bandpower(baseline_ERD(:,idx_C4),fs,[8 13])) / ...
             bandpower(baseline_ERD(:,idx_C4),fs,[8 13]);

    % BETA
    beta_C3 = (bandpower(trial_ERD(:,idx_C3),fs,[14 30]) - ...
               bandpower(baseline_ERD(:,idx_C3),fs,[14 30])) / ...
               bandpower(baseline_ERD(:,idx_C3),fs,[14 30]);

    beta_Cz = (bandpower(trial_ERD(:,idx_Cz),fs,[14 30]) - ...
               bandpower(baseline_ERD(:,idx_Cz),fs,[14 30])) / ...
               bandpower(baseline_ERD(:,idx_Cz),fs,[13 30]);

    beta_C4 = (bandpower(trial_ERD(:,idx_C4),fs,[14 30]) - ...
               bandpower(baseline_ERD(:,idx_C4),fs,[14 30])) / ...
               bandpower(baseline_ERD(:,idx_C4),fs,[14 30]);

    % LATERALIZACIÓN
    lat_mu = mu_C3 - mu_C4;
    lat_beta = beta_C3 - beta_C4;

    feat_ERD = [mu_C3 mu_Cz mu_C4 beta_C3 beta_Cz beta_C4 lat_mu lat_beta];

    %% -------- MRCP FEATURES --------
    % Cz simple (model 1)
    sigCz = trial_MRCP(:,idx_Cz);
    feat_MRCP_Cz = [min(sigCz), max(sigCz), trapz(sigCz), std(sigCz)];

    % All channels (model 2)
    sigC3 = trial_MRCP(:,idx_C3);
    sigC4 = trial_MRCP(:,idx_C4);

    feat_MRCP_all = [
        min(sigC3), min(sigCz), min(sigC4), ...
        max(sigC3), max(sigCz), max(sigC4), ...
        trapz(sigC3), trapz(sigCz), trapz(sigC4)
        ];
    %% COMBINE
    feat_model1 = [feat_ERD feat_MRCP_Cz];
    feat_model2 = [feat_ERD feat_MRCP_all];

    %% STORE
    switch EEG.EEG.event(ev).type
        case 'r'
            X_model1 = [X_model1; feat_model1];
            X_model2 = [X_model2; feat_model2];
            y = [y; "right"];

            MRCP_right = [MRCP_right; trial_MRCP(:,idx_Cz)'];

            % CSP ERD (beta + mu) 3 and 7 channels
            trial_count = trial_count + 1;

            trial_3ch = trial_ERD(:, [idx_C3 idx_Cz idx_C4]);
            trial_7ch = trial_ERD(:, [idx_C3 idx_Cz idx_C4 idx_P3 idx_P4 idx_F3 idx_F4]);

            X_trials_3ch(:,:,trial_count) = trial_3ch';
            X_trials_7ch(:,:,trial_count) = trial_7ch';

            % CSP mu 3 and 7 channels

            trial_3ch_mu = trial_mu(:, [idx_C3 idx_Cz idx_C4]);
            trial_7ch_mu = trial_mu(:, [idx_C3 idx_Cz idx_C4 idx_P3 idx_P4 idx_F3 idx_F4]);

            X_trials_3ch_mu(:,:,trial_count) = trial_3ch_mu';
            X_trials_7ch_mu(:,:,trial_count) = trial_7ch_mu';

            % CSP beta 3 and 7 channels

            trial_3ch_beta = trial_beta(:, [idx_C3 idx_Cz idx_C4]);
            trial_7ch_beta = trial_beta(:, [idx_C3 idx_Cz idx_C4 idx_P3 idx_P4 idx_F3 idx_F4]);

            X_trials_3ch_beta(:,:,trial_count) = trial_3ch_beta';
            X_trials_7ch_beta(:,:,trial_count) = trial_7ch_beta';

            y_trials(trial_count) = 1;

        case 'l'
            X_model1 = [X_model1; feat_model1];
            X_model2 = [X_model2; feat_model2];
            y = [y; "left"];

            MRCP_left = [MRCP_left; trial_MRCP(:,idx_Cz)'];

            % CSP ERD (beta + mu) 3 and 7 channels
            trial_count = trial_count + 1;

            trial_3ch = trial_ERD(:, [idx_C3 idx_Cz idx_C4]);
            trial_7ch = trial_ERD(:, [idx_C3 idx_Cz idx_C4 idx_P3 idx_P4 idx_F3 idx_F4]);

            X_trials_3ch(:,:,trial_count) = trial_3ch';
            X_trials_7ch(:,:,trial_count) = trial_7ch';

            % CSP mu 3 and 7 channels

            trial_3ch_mu = trial_mu(:, [idx_C3 idx_Cz idx_C4]);
            trial_7ch_mu = trial_mu(:, [idx_C3 idx_Cz idx_C4 idx_P3 idx_P4 idx_F3 idx_F4]);

            X_trials_3ch_mu(:,:,trial_count) = trial_3ch_mu';
            X_trials_7ch_mu(:,:,trial_count) = trial_7ch_mu';

            % CSP beta 3 and 7 channels

            trial_3ch_beta = trial_beta(:, [idx_C3 idx_Cz idx_C4]);
            trial_7ch_beta = trial_beta(:, [idx_C3 idx_Cz idx_C4 idx_P3 idx_P4 idx_F3 idx_F4]);

            X_trials_3ch_beta(:,:,trial_count) = trial_3ch_beta';
            X_trials_7ch_beta(:,:,trial_count) = trial_7ch_beta';

            y_trials(trial_count) = -1;

        case 'b'
            X_model1 = [X_model1; feat_model1];
            X_model2 = [X_model2; feat_model2];
            y = [y; "rest"];

            MRCP_rest = [MRCP_rest; trial_MRCP(:,idx_Cz)'];

            % CSP ERD (beta + mu) 3 and 7 channels
            trial_count = trial_count + 1;

            trial_3ch = trial_ERD(:, [idx_C3 idx_Cz idx_C4]);
            trial_7ch = trial_ERD(:, [idx_C3 idx_Cz idx_C4 idx_P3 idx_P4 idx_F3 idx_F4]);

            X_trials_3ch(:,:,trial_count) = trial_3ch';
            X_trials_7ch(:,:,trial_count) = trial_7ch';

            % CSP mu 3 and 7 channels

            trial_3ch_mu = trial_mu(:, [idx_C3 idx_Cz idx_C4]);
            trial_7ch_mu = trial_mu(:, [idx_C3 idx_Cz idx_C4 idx_P3 idx_P4 idx_F3 idx_F4]);

            X_trials_3ch_mu(:,:,trial_count) = trial_3ch_mu';
            X_trials_7ch_mu(:,:,trial_count) = trial_7ch_mu';

            % CSP beta 3 and 7 channels

            trial_3ch_beta = trial_beta(:, [idx_C3 idx_Cz idx_C4]);
            trial_7ch_beta = trial_beta(:, [idx_C3 idx_Cz idx_C4 idx_P3 idx_P4 idx_F3 idx_F4]);

            X_trials_3ch_beta(:,:,trial_count) = trial_3ch_beta';
            X_trials_7ch_beta(:,:,trial_count) = trial_7ch_beta';

            y_trials(trial_count) = 0;
    end
end

fprintf('Rejected trials: %d\n', rejected);

X_trials_3ch = X_trials_3ch(:,:,1:trial_count);
X_trials_7ch = X_trials_7ch(:,:,1:trial_count);

X_trials_3ch_mu = X_trials_3ch_mu(:,:,1:trial_count);
X_trials_3ch_beta = X_trials_3ch_beta(:,:,1:trial_count);

X_trials_7ch_mu = X_trials_7ch_mu(:,:,1:trial_count);
X_trials_7ch_beta = X_trials_7ch_beta(:,:,1:trial_count);

y_trials = y_trials(1:trial_count);
disp(unique(y_trials))

fprintf('Right: %d\n', sum(y_trials == 1));
fprintf('Left: %d\n', sum(y_trials == -1));
fprintf('Rest: %d\n', sum(y_trials == 0));

figure;

subplot(3,1,1)
histogram(std_C3,30)
title('STD C3')

subplot(3,1,2)
histogram(std_Cz,30)
title('STD Cz')

subplot(3,1,3)
histogram(std_C4,30)
title('STD C4')
%% ================= PLOTS =================
% Normalization for plots only
X1 = normalize(X_model1);
X2 = normalize(X_model2);

ev = first_trial*2;

start_1_trial = EEG.EEG.event(ev).latency;
end_1_trial = start_1_trial + 2*fs - 1;

trial_raw = signals(start_1_trial:end_1_trial, :);
trial_ERD_plot = data_ERD(start_1_trial:end_1_trial, :);
trial_MRCP_plot = data_MRCP(start_1_trial:end_1_trial, :);

% Raw vs Filtered ERD
figure;

subplot(2,1,1)
plot(trial_raw(:, idx_Cz))
title('1st trial Raw signal (Cz)')
xlabel('Samples');
ylabel('Amplitude')

subplot(2,1,2)
plot(trial_ERD_plot(:, idx_Cz))
title('1st ERD trial filtered(8-30 Hz)')
xlabel('Samples');
ylabel('Amplitude')

% Raw vs Filtered MRCP
figure;
subplot(2,1,1)
plot(trial_raw(:, idx_Cz))
title('1st trial Raw signal (Cz)')
xlabel('Samples');
ylabel('Amplitude')
subplot(2,1,2)
plot(trial_MRCP_plot(:, idx_Cz))
title('1st MRCP trial filtered(0.3-3 Hz)')
xlabel('Samples');
ylabel('Amplitude')


%% Mu lateralization Boxplot

figure;
boxplot(X1(:,7), y)
title('ERD Lateralization Mu (C3 - C4)')
ylabel('Normalized value')


%% MRCP min Boxplot

figure;
boxplot(X1(:,end-3), y)
title('MRCP Min (Cz)')
ylabel('Normalized value')


%% SCATTER ERD vs MRCP

figure;
gscatter(X1(:,7), X1(:,end-3), y)
xlabel('Lateralization Mu (C3 - C4)')
ylabel('MRCP Min (Cz)')
title('ERD vs MRCP Feature Space')


%% PCA 

[coeff, score] = pca(X2);

figure;
gscatter(score(:,1), score(:,2), y)
xlabel('PC1')
ylabel('PC2')
title('PCA Projection')


%% CORRELATION MATRIX 

figure;
imagesc(corr(X2))
colorbar
title('Feature Correlation Matrix (Model 2)')

%% Wavelet
figure;
cwt(trial_ERD(:,idx_C3), fs);
title('Wavelet Transform (C3)')
%% MRCP MEAN 

figure; hold on;
plot(mean(MRCP_left,1),'b','LineWidth',2)
plot(mean(MRCP_right,1),'r','LineWidth',2)
plot(mean(MRCP_rest,1),'g','LineWidth',2)
legend('Left','Right','Rest')
title('Mean MRCP (Cz)')
xlabel('Samples')
ylabel('Amplitude')

%% SAVE CSV

% Model 1: ERD (Cz C3 C4) features and MRCP (Cz) features
df1 = array2table(X_model1);
df1.label = y;
writetable(df1,'features_model1.csv');

% Model 2 ERD (Cz C3 C4) features and MRCP (Cz C3 C4) features
df2 = array2table(X_model2);
df2.label = y;
writetable(df2,'features_model2.csv');

% CSP
save('trials_CSP.mat',...
    'X_trials_3ch','X_trials_7ch',...
    'X_trials_3ch_mu','X_trials_3ch_beta',...
    'X_trials_7ch_mu','X_trials_7ch_beta',...
    'y_trials')
