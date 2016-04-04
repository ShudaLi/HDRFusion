function matlab2opencv_yml( resample, variableName, fileName, flag)
%refere
%http://answers.opencv.org/question/58084/import-matlab-matrix-to-opencv/

[rows, cols] = size(resample);

% Beware of Matlab's linear indexing
resample = resample';

% Write mode as default
if ( ~exist('flag','var') )
    flag = 'w'; 
end

if ( ~exist(fileName,'file') || flag == 'w' )
    % New file or write mode specified 
    file = fopen( fileName, 'w');
    fprintf( file, '%%YAML:1.0\n');
else
    % Append mode
    file = fopen( fileName, 'a');
end

% Write resample header
fprintf( file, '    %s: !!opencv-matrix\n', variableName);
fprintf( file, '        rows: %d\n', rows);
fprintf( file, '        cols: %d\n', cols);
fprintf( file, '        dt: f\n');
fprintf( file, '        data: [ ');

% Write resample data
for i=1:rows*cols
    fprintf( file, '%.6f', resample(i));
    if (i == rows*cols), break, end
    fprintf( file, ', ');
    if mod(i+1,4) == 0
        fprintf( file, '\n            ');
    end
end

fprintf( file, ']\n');
fclose(file);

end