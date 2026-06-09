%% START & CLEANUP
clearvars; close all; clc;
 
%% SUBJECTS
subjects = {'v01','v02','v03','v04','v05','v06','v07','v08','v09','v10','v11'};
 
%% GLOBAL ACCUMULATORS
all_features1 = [];
all_features2 = [];
all_labels = string([]);
all_subjects = string([]);
 
all_X3ch = [];
all_X8ch = [];
all_X3ch_mu = [];
all_X3ch_beta = [];
all_X8ch_mu = [];
all_X8ch_beta = [];
all_y_csp = [];
all_subj_csp  = [];

% Topomap global accumulators
all_topo_mu_left    = [];
all_topo_mu_right   = [];
all_topo_mu_rest    = [];
all_topo_beta_left  = [];
all_topo_beta_right = [];
all_topo_beta_rest  = [];
all_topo_mrcp_left  = [];
all_topo_mrcp_right = [];
all_topo_mrcp_rest  = [];
 
% Rejection log: [n_accepted, n_rejected] per subject
rej_log = zeros(numel(subjects), 2);
 
%% SUBJECT LOOP
for s = 1:numel(subjects)
 
    fprintf('\nProcessing %s\n', subjects{s});
 
    %% LOAD 
    filename = [subjects{s} '.mat'];
    EEG = load(filename);
 
    %% DEFINITIONS
    fs = EEG.EEG.srate;
    signals = detrend(EEG.EEG.data(1:15,:)', 'constant');
    num_events = numel(EEG.EEG.event);
    first_trial = 121;
 
    ch_names = {'F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','P7','P3','Pz','P4','P8'};
 
    %% CHANNEL INDEXES
    idx_C3 = find(strcmp(ch_names,'C3'));
    idx_Cz = find(strcmp(ch_names,'Cz'));
    idx_C4 = find(strcmp(ch_names,'C4'));
    idx_Fz = find(strcmp(ch_names,'Fz'));
    idx_P3 = find(strcmp(ch_names,'P3'));
    idx_P4 = find(strcmp(ch_names,'P4'));
    idx_F3 = find(strcmp(ch_names,'F3'));
    idx_F4 = find(strcmp(ch_names,'F4'));
 
    %% FILTERS
    [b_erd,  a_erd] = butter(4, [8  30]/(fs/2), 'bandpass');
    [b_mu,   a_mu] = butter(4, [8  13]/(fs/2), 'bandpass');
    [b_beta, a_beta] = butter(4, [14 30]/(fs/2), 'bandpass');
    [b_mrcp, a_mrcp] = butter(2, [0.3 3]/(fs/2), 'bandpass');
 
    %% FILTER SIGNALS
    data_ERD = filtfilt(b_erd,  a_erd,  signals);
    data_mu = filtfilt(b_mu,   a_mu,   signals);
    data_beta = filtfilt(b_beta, a_beta, signals);
    data_MRCP = filtfilt(b_mrcp, a_mrcp, signals);
 
    %% BAD CHANNEL DETECTION + CAR
    bad_channels = detect_bad_channels(data_ERD, fs);
    data_ERD = apply_car(data_ERD,  bad_channels);
    data_mu = apply_car(data_mu,   bad_channels);
    data_beta = apply_car(data_beta, bad_channels);
    data_MRCP = apply_car(data_MRCP, bad_channels);
 
    %% STORE FIRST SUBJECT DATA FOR EDA PLOTS
    if s == 1
        ev_first  = first_trial * 2;
        t0  = EEG.EEG.event(ev_first).latency;
        raw_plot = signals(t0 : t0 + 2*fs - 1, :);
        erd_plot = data_ERD(t0 : t0 + 2*fs - 1, :);
        mrcp_plot = data_MRCP(t0 : t0 + 2*fs - 1, :);
        fs_plot = fs;
    end
 
%% CALCULATE ADAPTATIVE THRESHOKDS FOR TRIAL REJECTION (95% PERCENTILE)
    all_max_amp = [];
    all_std_C3 = [];
    all_std_Cz = [];
    all_std_C4 = [];
 
    for ev_tmp = first_trial*2 : 2 : num_events
        st = EEG.EEG.event(ev_tmp).latency;
        sp = st + 2*fs - 1;
        if sp > size(signals,1) 
            continue;
        end
 
        win_tmp = round(0.5*fs) : round(1.5*fs);
        t_tmp = detrend(data_ERD(st:sp, :), 'constant');
        t_tmp = t_tmp(win_tmp, :);
 
        all_max_amp(end+1) = max(abs(t_tmp(:, [idx_C3 idx_Cz idx_C4])), [], 'all'); 
        all_std_C3(end+1) = std(t_tmp(:, idx_C3)); 
        all_std_Cz(end+1) = std(t_tmp(:, idx_Cz)); 
        all_std_C4(end+1) = std(t_tmp(:, idx_C4)); 
    end
 
    thr_amp = prctile(all_max_amp, 95);
    thr_std = prctile([all_std_C3, all_std_Cz, all_std_C4], 95);
 
    fprintf('Adaptive thresholds: amp p95: %.2f uV  std p95: %.2f uV\n', thr_amp, thr_std);
 
    %% FIGURES TO JUSTIFY TRIAL REJECTION
    figure();
 
    subplot(2,1,1);
    histogram(all_max_amp, 30); 
    hold on;
    xline(thr_amp, 'r--', 'LineWidth', 2);
    title(sprintf('%s - Max amplitude per trial (C3/Cz/C4)', subjects{s}));
    xlabel('Amplitude (uV)'); ylabel('Count');
    legend('Trials', sprintf('p95 = %.1f uV', thr_amp));
 
    subplot(2,1,2);
    histogram([all_std_C3, all_std_Cz, all_std_C4], 30); 
    hold on;
    xline(thr_std, 'r--', 'LineWidth', 2);
    title(sprintf('%s - Std per trial (C3/Cz/C4)', subjects{s}));
    xlabel('Std (uV)'); ylabel('Count');
    legend('Trials', sprintf('p95 = %.2f uV', thr_std));
 
    sgtitle(sprintf('Trial rejection threshold - %s', subjects{s}));
 
    %% EXTRACTING EPOCHS
    X_model1 = [];
    X_model2 = [];
    y_subj = string([]);
 
    X_trials_3ch = [];
    X_trials_8ch = [];
    X_trials_3ch_mu = [];
    X_trials_3ch_beta = [];
    X_trials_8ch_mu = [];
    X_trials_8ch_beta = [];
    y_trials_subj = [];

    % Topomap accumulators per subject
    topo_mu_left = [];
    topo_mu_right = [];
    topo_mu_rest = [];
    topo_beta_left = [];
    topo_beta_right = [];
    topo_beta_rest = [];
    topo_mrcp_left = [];
    topo_mrcp_right = [];
    topo_mrcp_rest = [];
 
    rejected = 0;
    trial_count = 0;
 
    for ev = first_trial*2 : 2 : num_events
 
        start_sample = EEG.EEG.event(ev).latency;
        stop_sample = start_sample + 2*fs - 1;
        baseline_start = start_sample - 2*fs;
        baseline_end = start_sample - 1;
        mrcp_start = start_sample - round(0.5*fs);
 
        % Skip out-of-bounds
        if stop_sample > size(signals,1) || baseline_start < 1 || mrcp_start < 1
            continue;
        end
 
        trial_ERD = data_ERD(start_sample:stop_sample, :);
        trial_MRCP = data_MRCP(start_sample:stop_sample, :);
        baseline_ERD = data_ERD(baseline_start:baseline_end, :);
        trial_mu = data_mu(start_sample:stop_sample, :);
        trial_beta = data_beta(start_sample:stop_sample, :);
        baseline_mu = data_mu(baseline_start:baseline_end, :);
        baseline_beta = data_beta(baseline_start:baseline_end, :);
 
        %% Detrend epochs
        trial_ERD = detrend(trial_ERD, 'constant');
        trial_MRCP = detrend(trial_MRCP, 'constant');
        baseline_ERD = detrend(baseline_ERD, 'constant');
        trial_mu = detrend(trial_mu, 'constant');
        trial_beta = detrend(trial_beta, 'constant');
        baseline_mu = detrend(baseline_mu, 'constant');
        baseline_beta = detrend(baseline_beta, 'constant');
 
        %% Relevant window: ERD 0.5 s - 1.5 s / MRCP -0.5 s - 0.5 s
        win = round(0.5*fs) : round(1.5*fs);
        trial_ERD = trial_ERD(win, :);
        trial_mu = trial_mu(win, :);
        trial_beta = trial_beta(win, :);

        onset_idx = fs + 1;
        win_MRCP = (onset_idx - round(0.5*fs)) : (onset_idx + round(0.5*fs));
        trial_MRCP = detrend(trial_MRCP(win_MRCP, :), 'constant');
 
        %% TRIAL REJECTION WITH ADAPTATIVE THRESHOLD (PERCENTILE 95)
        bad = 0;
        sigC3 = trial_ERD(:, idx_C3);
        sigCz = trial_ERD(:, idx_Cz);
        sigC4 = trial_ERD(:, idx_C4);
 
        if max(abs(sigC3)) > thr_amp || std(sigC3) > thr_std
            bad = bad + 1;
        end
        if max(abs(sigCz)) > thr_amp || std(sigCz) > thr_std
            bad = bad + 1;
        end
        if max(abs(sigC4)) > thr_amp || std(sigC4) > thr_std
            bad = bad + 1;
        end
 
        if bad >= 2
            rejected = rejected + 1;
            continue;
        end
 
        %%  ERD FEATURES + TOPOMAP DATA (all 15 channels)
        % Compute bandpower for all channels once — avoids redundant calls
        % Features for C3/Cz/C4 are extracted directly from the topomap arrays
        topo_mu_trial   = zeros(1, 15);
        topo_beta_trial = zeros(1, 15);

        for ch = 1:15
            bp_mu_t  = bandpower(trial_ERD(:,ch),    fs, [8  13]);
            bp_mu_b  = bandpower(baseline_ERD(:,ch), fs, [8  13]);
            bp_be_t  = bandpower(trial_ERD(:,ch),    fs, [14 30]);
            bp_be_b  = bandpower(baseline_ERD(:,ch), fs, [14 30]);
            if bp_mu_b  > 0, topo_mu_trial(ch)   = (bp_mu_t  - bp_mu_b)  / bp_mu_b;  end
            if bp_be_b  > 0, topo_beta_trial(ch) = (bp_be_t  - bp_be_b)  / bp_be_b;  end
        end

        % Extract C3, Cz, C4 directly from topomap arrays (no redundant calculation)
        mu_C3   = topo_mu_trial(idx_C3);
        mu_Cz   = topo_mu_trial(idx_Cz);
        mu_C4   = topo_mu_trial(idx_C4);
        beta_C3 = topo_beta_trial(idx_C3);
        beta_Cz = topo_beta_trial(idx_Cz);
        beta_C4 = topo_beta_trial(idx_C4);

        % Lateralization (C3 - C4)
        lat_mu   = mu_C3 - mu_C4;
        lat_beta = beta_C3 - beta_C4;

        feat_ERD = [mu_C3, mu_Cz, mu_C4, beta_C3, beta_Cz, beta_C4, lat_mu, lat_beta];
 
        %%  MRCP FEATURES
 
        % MRCP topomap: mean amplitude across time for all 15 channels
        topo_mrcp_trial = mean(trial_MRCP, 1);   % 1 x 15

        % Model 1: Cz + Fz
        sigCz_mrcp   = trial_MRCP(:, idx_Cz);
        sigFz_mrcp   = trial_MRCP(:, idx_Fz);
        feat_MRCP_Cz = [min(sigCz_mrcp), max(sigCz_mrcp), trapz(sigCz_mrcp), std(sigCz_mrcp), ...
                        min(sigFz_mrcp), max(sigFz_mrcp), trapz(sigFz_mrcp), std(sigFz_mrcp)];
 
        % Model 2: C3 + Cz + C4 + Fz
        sigC3_mrcp = trial_MRCP(:, idx_C3);
        sigC4_mrcp = trial_MRCP(:, idx_C4);
 
        feat_MRCP_all = [min(sigC3_mrcp),  min(sigCz_mrcp),  min(sigC4_mrcp),  min(sigFz_mrcp), ...
                         max(sigC3_mrcp),  max(sigCz_mrcp),  max(sigC4_mrcp),  max(sigFz_mrcp), ...
                         trapz(sigC3_mrcp),trapz(sigCz_mrcp),trapz(sigC4_mrcp),trapz(sigFz_mrcp), ...
                         std(sigC3_mrcp),  std(sigCz_mrcp),  std(sigC4_mrcp),  std(sigFz_mrcp)];
 
        %% Combine features
        feat_model1 = [feat_ERD, feat_MRCP_Cz];
        feat_model2 = [feat_ERD, feat_MRCP_all];
 
        %% STORE BY CLASS
        event_type = EEG.EEG.event(ev).type;
        if ~ismember(event_type, {'r','l','b'})
            continue;
        end
 
        X_model1 = [X_model1; feat_model1]; 
        X_model2 = [X_model2; feat_model2]; 
 
        switch event_type
            case 'r'
                y_subj(end+1,1) = "right";
                topo_mu_right   = [topo_mu_right;   topo_mu_trial];
                topo_beta_right = [topo_beta_right;  topo_beta_trial];
                topo_mrcp_right = [topo_mrcp_right;  topo_mrcp_trial];
            case 'l'
                y_subj(end+1,1) = "left";
                topo_mu_left    = [topo_mu_left;    topo_mu_trial];
                topo_beta_left  = [topo_beta_left;  topo_beta_trial];
                topo_mrcp_left  = [topo_mrcp_left;  topo_mrcp_trial];
            case 'b'
                y_subj(end+1,1) = "rest";
                topo_mu_rest    = [topo_mu_rest;    topo_mu_trial];
                topo_beta_rest  = [topo_beta_rest;  topo_beta_trial];
                topo_mrcp_rest  = [topo_mrcp_rest;  topo_mrcp_trial];
        end
 
        %% CSP TRIALS
        trial_count = trial_count + 1;
 
        trial_3ch = trial_ERD(:, [idx_C3, idx_Cz, idx_C4]);
        trial_8ch = trial_ERD(:, [idx_C3, idx_Cz, idx_C4, idx_P3, idx_P4, idx_F3, idx_F4, idx_Fz]);
 
        trial_3ch_mu = trial_mu(:, [idx_C3, idx_Cz, idx_C4]);
        trial_8ch_mu = trial_mu(:, [idx_C3, idx_Cz, idx_C4, idx_P3, idx_P4, idx_F3, idx_F4, idx_Fz]);
        trial_3ch_beta = trial_beta(:, [idx_C3, idx_Cz, idx_C4]);
        trial_8ch_beta = trial_beta(:, [idx_C3, idx_Cz, idx_C4, idx_P3, idx_P4, idx_F3, idx_F4, idx_Fz]);
 
        X_trials_3ch(:,:,trial_count) = trial_3ch';   
        X_trials_8ch(:,:,trial_count) = trial_8ch';     
        X_trials_3ch_mu(:,:,trial_count) = trial_3ch_mu';  
        X_trials_8ch_mu(:,:,trial_count) = trial_8ch_mu';   
        X_trials_3ch_beta(:,:,trial_count) = trial_3ch_beta'; 
        X_trials_8ch_beta(:,:,trial_count) = trial_8ch_beta';
 
        switch event_type
            case 'r'
                y_trials_subj(trial_count) =  1;
            case 'l'
                y_trials_subj(trial_count) = -1;
            case 'b'
                y_trials_subj(trial_count) =  0;
        end
 
    end
 
    %% REJECTION SUMMARY
    n_total = trial_count + rejected;
    rej_log(s,:) = [trial_count, rejected];
    fprintf('Accepted: %d / %d  (%.1f%% rejected)\n', trial_count, n_total, 100*rejected/n_total);
    fprintf('Right: %d | Left: %d | Rest: %d\n', sum(y_trials_subj == 1), sum(y_trials_subj == -1),sum(y_trials_subj == 0));
 
    %% ACCUMULATE FEATURES
    subj_col  = repmat(string(subjects{s}), size(X_model1,1), 1);
 
    all_features1 = [all_features1; X_model1];
    all_features2 = [all_features2; X_model2]; 
    all_labels    = [all_labels;    y_subj]; 
    all_subjects  = [all_subjects;  subj_col];
 
    %% ACCUMULATE CSP TRIALS
    subj_idx_vec = repmat(s, 1, trial_count);
 
    if isempty(all_X3ch)
        all_X3ch = X_trials_3ch;
        all_X8ch = X_trials_8ch;
        all_X3ch_mu = X_trials_3ch_mu;
        all_X3ch_beta = X_trials_3ch_beta;
        all_X8ch_mu = X_trials_8ch_mu;
        all_X8ch_beta = X_trials_8ch_beta;
    else
        all_X3ch = cat(3, all_X3ch, X_trials_3ch);
        all_X8ch = cat(3, all_X8ch, X_trials_8ch);
        all_X3ch_mu = cat(3, all_X3ch_mu, X_trials_3ch_mu);
        all_X3ch_beta = cat(3, all_X3ch_beta, X_trials_3ch_beta);
        all_X8ch_mu = cat(3, all_X8ch_mu, X_trials_8ch_mu);
        all_X8ch_beta = cat(3, all_X8ch_beta, X_trials_8ch_beta);
    end
 
    all_y_csp    = [all_y_csp, y_trials_subj];
    all_subj_csp = [all_subj_csp, subj_idx_vec]; 

    %% ACCUMULATE TOPOMAP DATA ACROSS SUBJECTS
    if s == 1
        all_topo_mu_left    = topo_mu_left;
        all_topo_mu_right   = topo_mu_right;
        all_topo_mu_rest    = topo_mu_rest;
        all_topo_beta_left  = topo_beta_left;
        all_topo_beta_right = topo_beta_right;
        all_topo_beta_rest  = topo_beta_rest;
        all_topo_mrcp_left  = topo_mrcp_left;
        all_topo_mrcp_right = topo_mrcp_right;
        all_topo_mrcp_rest  = topo_mrcp_rest;
    else
        all_topo_mu_left    = [all_topo_mu_left;    topo_mu_left];
        all_topo_mu_right   = [all_topo_mu_right;   topo_mu_right];
        all_topo_mu_rest    = [all_topo_mu_rest;     topo_mu_rest];
        all_topo_beta_left  = [all_topo_beta_left;   topo_beta_left];
        all_topo_beta_right = [all_topo_beta_right;  topo_beta_right];
        all_topo_beta_rest  = [all_topo_beta_rest;   topo_beta_rest];
        all_topo_mrcp_left  = [all_topo_mrcp_left;   topo_mrcp_left];
        all_topo_mrcp_right = [all_topo_mrcp_right;  topo_mrcp_right];
        all_topo_mrcp_rest  = [all_topo_mrcp_rest;   topo_mrcp_rest];
    end
 
end 
 
%%  EDA PLOTS - 1ST SUBJECT DATA
 
t_ax = (0 : size(raw_plot,1)-1) / fs_plot;
 
%% Raw vs filtered ERD
figure;
subplot(2,1,1);
plot(t_ax, raw_plot(:, idx_Cz));
title('1st trial - Raw signal (Cz)');
xlabel('Time (s)'); ylabel('Amplitude (uV)');
 
subplot(2,1,2);
plot(t_ax, erd_plot(:, idx_Cz));
title('1st trial - ERD filtered (8-30 Hz)');
xlabel('Time (s)'); ylabel('Amplitude (uV)');
sgtitle('Raw vs filtered ERD - v01');
 
%% Raw vs filtered MRCP
figure;
subplot(2,1,1);
plot(t_ax, raw_plot(:, idx_Cz));
title('1st trial - Raw signal (Cz)');
xlabel('Time (s)'); ylabel('Amplitude (uV)');
 
subplot(2,1,2);
plot(t_ax, mrcp_plot(:, idx_Cz));
title('1st trial - MRCP filtered (0.3-3 Hz)');
xlabel('Time (s)'); ylabel('Amplitude (uV)');
sgtitle('Raw vs filtered MRCP - v01');

%% TOPOGRAPHIC MAPS — grand average all subjects
% Load channel locations for this dataset
chanlocs = readlocs('coord_15elecMotorImagery.loc');

% Grand means per class
mu_left_mean    = mean(all_topo_mu_left,    1);
mu_right_mean   = mean(all_topo_mu_right,   1);
mu_rest_mean    = mean(all_topo_mu_rest,    1);
beta_left_mean  = mean(all_topo_beta_left,  1);
beta_right_mean = mean(all_topo_beta_right, 1);
beta_rest_mean  = mean(all_topo_beta_rest,  1);
mrcp_left_mean  = mean(all_topo_mrcp_left,  1);
mrcp_right_mean = mean(all_topo_mrcp_right, 1);
mrcp_rest_mean  = mean(all_topo_mrcp_rest,  1);
mrcp_move_mean  = mean([all_topo_mrcp_left; all_topo_mrcp_right], 1);

%% ERD Mu topomaps
figure('Position', [100 100 1400 400]);
lim_mu = max(abs([mu_left_mean, mu_right_mean, mu_rest_mean]));

subplot(1,4,1);
topoplot(mu_left_mean,  chanlocs, 'electrodes','labels','maplimits',[-lim_mu lim_mu]);
colorbar; title('Mu ERD - Left');

subplot(1,4,2);
topoplot(mu_right_mean, chanlocs, 'electrodes','labels','maplimits',[-lim_mu lim_mu]);
colorbar; title('Mu ERD - Right');

subplot(1,4,3);
topoplot(mu_rest_mean,  chanlocs, 'electrodes','labels','maplimits',[-lim_mu lim_mu]);
colorbar; title('Mu ERD - Rest');

subplot(1,4,4);
topoplot(mu_left_mean - mu_right_mean, chanlocs, 'electrodes','labels','maplimits','absmax');
colorbar; title('Mu ERD - Left minus Right');

sgtitle('Mu ERD Topographic Maps (8-13 Hz) - Grand average all subjects');

%% ERD Beta topomaps
figure('Position', [100 100 1400 400]);
lim_beta = max(abs([beta_left_mean, beta_right_mean, beta_rest_mean]));

subplot(1,4,1);
topoplot(beta_left_mean,  chanlocs, 'electrodes','labels','maplimits',[-lim_beta lim_beta]);
colorbar; title('Beta ERD - Left');

subplot(1,4,2);
topoplot(beta_right_mean, chanlocs, 'electrodes','labels','maplimits',[-lim_beta lim_beta]);
colorbar; title('Beta ERD - Right');

subplot(1,4,3);
topoplot(beta_rest_mean,  chanlocs, 'electrodes','labels','maplimits',[-lim_beta lim_beta]);
colorbar; title('Beta ERD - Rest');

subplot(1,4,4);
topoplot(beta_left_mean - beta_right_mean, chanlocs, 'electrodes','labels','maplimits','absmax');
colorbar; title('Beta ERD - Left minus Right');

sgtitle('Beta ERD Topographic Maps (14-30 Hz) - Grand average all subjects');

%% MRCP topomaps
figure('Position', [100 100 1400 400]);
lim_mrcp = max(abs([mrcp_left_mean, mrcp_right_mean, mrcp_rest_mean]));

subplot(1,4,1);
topoplot(mrcp_left_mean,  chanlocs, 'electrodes','labels','maplimits',[-lim_mrcp lim_mrcp]);
colorbar; title('MRCP - Left');

subplot(1,4,2);
topoplot(mrcp_right_mean, chanlocs, 'electrodes','labels','maplimits',[-lim_mrcp lim_mrcp]);
colorbar; title('MRCP - Right');

subplot(1,4,3);
topoplot(mrcp_rest_mean,  chanlocs, 'electrodes','labels','maplimits',[-lim_mrcp lim_mrcp]);
colorbar; title('MRCP - Rest');

subplot(1,4,4);
topoplot(mrcp_move_mean - mrcp_rest_mean, chanlocs, 'electrodes','labels','maplimits','absmax');
colorbar; title('MRCP - Movement minus Rest');

sgtitle('MRCP Topographic Maps (0.3-3 Hz) - Grand average all subjects');
 
%%  REJECTION SUMMARY
figure;
bar_data = [rej_log(:,1), rej_log(:,2)];
bar(bar_data, 'stacked');
legend('Accepted','Rejected');
xticks(1:numel(subjects));
xticklabels(subjects);
ylabel('Number of trials');
title('Trial rejection per subject');
grid on;
 
% Rejection rate per subject
rej_rate = 100 * rej_log(:,2) ./ sum(rej_log, 2);
fprintf('\nRejection rate per subject:\n');
for s = 1:numel(subjects)
    fprintf('%s: %.1f%%\n', subjects{s}, rej_rate(s));
end
fprintf('Mean: %.1f%% +/- %.1f%%\n', mean(rej_rate), std(rej_rate));
 
%%  FEATURE ANALYSIS
% M2: 20 features (ERD + MRCP C3+Cz+C4)
feature_names_m2 = {'mu\_C3','mu\_Cz','mu\_C4', ...
                    'beta\_C3','beta\_Cz','beta\_C4', ...
                    'lat\_mu','lat\_beta', ...
                    'mrcp\_min\_C3','mrcp\_min\_Cz','mrcp\_min\_C4','mrcp\_min\_Fz', ...
                    'mrcp\_max\_C3','mrcp\_max\_Cz','mrcp\_max\_C4','mrcp\_max\_Fz', ...
                    'mrcp\_auc\_C3','mrcp\_auc\_Cz','mrcp\_auc\_C4','mrcp\_auc\_Fz', ...
                    'mrcp\_std\_C3','mrcp\_std\_Cz','mrcp\_std\_C4','mrcp\_std\_Fz'};
 
% Normalize for visualization only
X1_norm = normalize(all_features1);
X2_norm = normalize(all_features2);
 
%% INDEX MASKS
idx_left = strcmp(all_labels, 'left');
idx_right = strcmp(all_labels, 'right');
idx_rest = strcmp(all_labels, 'rest');
idx_move = idx_left | idx_right; 
 
%% KRUSKAL-WALLIS: rest vs movement 
p_rm = zeros(1, size(all_features2, 2));
for f = 1:size(all_features2, 2)
    labels_rm          = all_labels;
    labels_rm(idx_move) = "move";
    p_rm(f) = kruskalwallis(all_features2(:,f), labels_rm, 'off');
end
 
%% KRUSKAL-WALLIS: left vs right
p_lr = zeros(1, size(all_features2, 2));
for f = 1:size(all_features2, 2)
    x_move   = all_features2(idx_move, f);
    lab_move = all_labels(idx_move);
    p_lr(f)  = kruskalwallis(x_move, lab_move, 'off');
end
 
%% COHEN'S D: rest vs movement
d_rm = zeros(1, size(all_features2, 2));
for f = 1:size(all_features2, 2)
    x_r = all_features2(idx_rest, f);
    x_m = all_features2(idx_move, f);
    pooled_std = sqrt((var(x_r) + var(x_m)) / 2);
    if pooled_std > 0
        d_rm(f) = abs(mean(x_r) - mean(x_m)) / pooled_std;
    end
end
 
%% COHEN'S D: left vs right
d_lr = zeros(1, size(all_features2, 2));
for f = 1:size(all_features2, 2)
    x_l = all_features2(idx_left,  f);
    x_r = all_features2(idx_right, f);
    pooled_std = sqrt((var(x_l) + var(x_r)) / 2);
    if pooled_std > 0
        d_lr(f) = abs(mean(x_l) - mean(x_r)) / pooled_std;
    end
end
 
%% FEATURE STATISTICAL ANALYSIS FIGURES
figure;
 
subplot(2,2,1);
bar(-log10(p_rm));
yline(-log10(0.05), 'r--', 'LineWidth', 1.5);
xticks(1:numel(feature_names_m2));
xticklabels(feature_names_m2);
xtickangle(90);
ylabel('-log_{10}(p-value)');
title('Kruskal-Wallis - rest vs movement');
legend('Features','p = 0.05','Location','northwest');
grid on;
 
subplot(2,2,2);
bar(-log10(p_lr));
yline(-log10(0.05), 'r--', 'LineWidth', 1.5);
xticks(1:numel(feature_names_m2));
xticklabels(feature_names_m2);
xtickangle(90);
ylabel('-log_{10}(p-value)');
title('Kruskal-Wallis - left vs right');
legend('Features','p = 0.05','Location','northeast');
grid on;
 
subplot(2,2,3);
bar(d_rm);
yline(0.2, 'r--', 'LineWidth', 1.5);
xticks(1:numel(feature_names_m2));
xticklabels(feature_names_m2);
xtickangle(90);
ylabel("Cohen's d");
title("Cohen's - rest vs movement");
legend('Features','d = 0.02','Location','northwest');
grid on;
 
subplot(2,2,4);
bar(d_lr);
yline(0.2, 'r--', 'LineWidth', 1.5);
xticks(1:numel(feature_names_m2));
xticklabels(feature_names_m2);
xtickangle(90);
ylabel("Cohen's d");
title("Cohen's - left vs right");
legend('Features','d = 0.2','Location','northeast');
grid on;
 
sgtitle('Feature Statistical Analysis: rest vs movement  |  left vs right');

%% MRCP min Fz boxplot - most discriminative MRCP feature
figure;
boxplot(X1_norm(:,13), all_labels);  
title('MRCP min (Fz, normalized)');
ylabel('Normalized value');

%% Lateralization mu - left vs right
figure;
idx_lr = idx_left | idx_right;
boxplot(X1_norm(idx_lr, 7), all_labels(idx_lr));
title('ERD lateralization mu - left vs right (normalized)');
ylabel('Normalized value');

%% Scatter MRCP min Fz vs Lateralization mu
figure;
gscatter(X1_norm(:,7), X1_norm(:,13), all_labels);
xlabel('Lateralization mu');
ylabel('MRCP min (Fz)');
title('Best ERD feature vs best MRCP feature');
legend('Location','best');
%% PCA
[~, score] = pca(X2_norm);
figure;
gscatter(score(:,1), score(:,2), all_labels);
xlabel('PC1'); ylabel('PC2');
title('PCA projection - Model 2 features (all subjects)');
 
%% Correlation matrix
figure;
imagesc(corr(all_features2));
colorbar;
xticks(1:numel(feature_names_m2)); 
xticklabels(feature_names_m2); 
xtickangle(45);
yticks(1:numel(feature_names_m2));
yticklabels(feature_names_m2);
title('Feature correlation matrix - Model 2');
 
%% SAVE OUTPUTS 
%% CSP .mat
save('trials_CSP_all.mat', ...
    'all_X3ch',      'all_X8ch', ...
    'all_X3ch_mu',   'all_X3ch_beta', ...
    'all_X8ch_mu',   'all_X8ch_beta', ...
    'all_y_csp',     'all_subj_csp');
 
%% FEATURE CSV FOR MODEL 1 AND MODEL 2
col_names_m1 = {'mu_C3','mu_Cz','mu_C4', ...
                'beta_C3','beta_Cz','beta_C4', ...
                'lat_mu','lat_beta', ...
                'mrcp_min_Cz','mrcp_max_Cz','mrcp_auc_Cz','mrcp_std_Cz', ...
                'mrcp_min_Fz','mrcp_max_Fz','mrcp_auc_Fz','mrcp_std_Fz'};
 
col_names_m2 = {'mu_C3','mu_Cz','mu_C4', ...
                'beta_C3','beta_Cz','beta_C4', ...
                'lat_mu','lat_beta', ...
                'mrcp_min_C3','mrcp_min_Cz','mrcp_min_C4','mrcp_min_Fz', ...
                'mrcp_max_C3','mrcp_max_Cz','mrcp_max_C4','mrcp_max_Fz', ...
                'mrcp_auc_C3','mrcp_auc_Cz','mrcp_auc_C4','mrcp_auc_Fz', ...
                'mrcp_std_C3','mrcp_std_Cz','mrcp_std_C4','mrcp_std_Fz'};
 
df1 = array2table(all_features1, 'VariableNames', col_names_m1);
df1.label = all_labels;
df1.subject = all_subjects;
writetable(df1, 'features_model1_all.csv');
 
df2 = array2table(all_features2, 'VariableNames', col_names_m2);
df2.label = all_labels;
df2.subject = all_subjects;
writetable(df2, 'features_model2_all.csv');