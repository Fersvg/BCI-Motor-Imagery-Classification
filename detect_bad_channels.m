function bad_channels = detect_bad_channels(data, fs)

% Detect bad EEG channels based on statistical outliers across segments

num_channels = size(data,1);
n_samples = size(data,2);

% Define sliding window (2 s with 50% overlap)

segment_length = fs * 2;

step = segment_length / 2;

starts = 1:step:(n_samples - segment_length + 1);

n_segments = length(starts);

bad_count = zeros(num_channels, 1);

for s = 1:n_segments
    
    % Extract current segment
    segment = data(:, starts(s):starts(s)+segment_length-1);
    
    % Channel correlation 
    R = corrcoef(segment');
    
    ch_corr = zeros(num_channels, 1);
    for i = 1:num_channels
        ch_corr(i) = mean(R(i, [1:i-1 i+1:num_channels]));
    end
    
    % Statistical features per channel

    ch_var = var(segment, 0, 2);
    ch_amp = max(segment, [],2) - min(segment, [], 2);
    ch_kurt = kurtosis(segment,0,2);
    
    % Normalize features (z-score)

    z_corr = zscore(ch_corr);
    z_var  = zscore(ch_var);
    z_amp  = zscore(ch_amp);
    z_kurt = zscore(ch_kurt);
    
    % Outlier threshold
    z_th = 3;
    
    % Channels flagged as bad in this segment
    bad_seg = unique([find(abs(z_corr)>z_th); find(abs(z_var)>z_th); find(abs(z_amp)>z_th); find(abs(z_kurt)>z_th)]);
    
    % Accumulate bad detections
    bad_count(bad_seg) = bad_count(bad_seg) + 1;
end

% Final decision: bad if flagged in >50% of segments
bad_channels = find(bad_count > n_segments/2);

end