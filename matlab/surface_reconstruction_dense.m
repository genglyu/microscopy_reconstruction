file_path = '/home/lvgeng/Code/TestingData/robotic/matlab/testing_pose.json';
robotic_pose_list = jsondecode(fileread(file_path));
points = robotic_pose_list(:,[13,14,15])

x = points(:, 1)
y = points(:, 2)
z = points(:, 3)



%% Fit: 'untitled fit 1'.
[xData, yData, zData] = prepareSurfaceData( x, y, z );

% Set up fittype and options.
ft = fittype( 'loess' );
opts = fitoptions( 'Method', 'LowessFit' );
opts.Normalize = 'on';
opts.Robust = 'Bisquare';
opts.Span = 0.7;

% Fit model to data.
[fitresult, gof] = fit( [xData, yData], zData, ft, opts );

%% extract data
[x2, y2] = meshgrid(min(x):0.0005:max(x), min(y):0.0005:max(y))
z2 = [fitresult(x2, y2)]

generated_points = [reshape(x2, [], 1), reshape(y2, [], 1), reshape(z2, [], 1)]


file_path = '/home/lvgeng/Code/TestingData/robotic/matlab/matlab.json';

jsonStr = jsonencode(generated_points);
fid = fopen(file_path, 'w');
if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, jsonStr, 'char');
fclose(fid);

plot( fitresult, [xData, yData], zData );
plot( fitresult, [reshape(x2, [], 1), reshape(y2, [], 1)], reshape(z2, [], 1) );
plot( fitresult, [xData, yData], zData );