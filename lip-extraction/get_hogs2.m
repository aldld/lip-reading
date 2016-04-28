function [] = get_hogs2(data_dir, single_folder)
    speakers = dir(data_dir);
   
    tic
    if nargin == 1
        parfor j = 1:numel(speakers)
            lip_folder = speakers(j).name;

            lip_folder_path = [data_dir lip_folder filesep];
            hog_folder_path = [lip_folder_path 'hog' filesep];

            folder_helper(lip_folder_path, hog_folder_path);
        end
    else
        dir_out = [data_dir 'hog' filesep];
        folder_helper(data_dir, dir_out, 1);
    end
    toc

end

function [] = folder_helper(dir_in, dir_out, par)
    lips = dir([dir_in '*.mat']);
    
    tic 
    if nargin == 2
        for j = 1:numel(lips)
            lip_name = lips(j).name;    
            lip_path = [dir_in lip_name];
            hog_name = strsplit(lip_name, '.');
            hog_path = [dir_out char(hog_name(1)) '.mat'];
            file_helper(lip_path, hog_path);
        end
    else
        parfor j = 1:numel(lips)
            lip_name = lips(j).name;    
            lip_path = [dir_in lip_name];
            hog_name = strsplit(lip_name, '.');
            hog_path = [dir_out char(hog_name(1)) '.mat'];
            file_helper(lip_path, hog_path);
        end
    end
    toc
end

function [] = file_helper(file_in, file_out)    
    mouths = load(file_in);
    mouths = mouths.mouths;
    
    cell_size = 10;
    
    hogs = [];
    
    for i = 1:size(mouths, 1)
        m = squeeze(mouths(i, :, :, :));
        try
            hog = vl_hog(single(m), cell_size);
        catch
            sprintf('File %s failed at frame %i', file_in, i);
            continue;
        end
        hogs(i, :, :, :) = hog;
    end

    save(file_out, 'hogs');
end
