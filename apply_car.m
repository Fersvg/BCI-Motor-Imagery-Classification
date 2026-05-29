function data = apply_car(data, bad_channels)

% Common Average Reference excluding bad channels

all_ch = 1:size(data, 1);

good_channels = setdiff(all_ch, bad_channels);

global_avg = mean(data(good_channels, :), 1);

data = data - global_avg;

end