% Compute hogs for all files in the given directory.

lip_folder = '/Users/eric/Programming/prog_crs/lip-reading/lip-extraction/lips/';
hogs_folder = '/Users/eric/Programming/prog_crs/lip-reading/lip-extraction/hogs/';

files = dir([lip_folder '*.mat']);

tic
parfor ii = 1:length(files)
    file_in = [lip_folder files(ii).name];
    file_out = [hogs_folder files(ii).name];
    
    mouths_to_hog(file_in, file_out);
end
toc
