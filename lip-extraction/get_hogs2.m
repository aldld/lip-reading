function [] = get_hogs2(data_dir)
    speakers = dir([data_dir 's*']);
    
    for j = 1:numel(speakers)
        lip_folder = speakers(j).name;
        lip_folder_path = [data_dir lip_folder];
        hog_folder_path = [lip_folder_path 'hog'];
        folder_helper();
    end

end

function [] = folder_helper(dir_in, dir_out)
    lips = dir([dir_in '*.mat']);
    
    for j = 1:numel(lips)
        lip_name = aligns(j).name;    
        lip_path = [dir_in lip_name];
        hog_name = char(strsplit(lip_name, '.'));
        hog_path = [dir_out hog_name(1) '.mat'];
        tic
        filehelper(lip_path, hog_path);
        toc
    end
end

function [] = file_helper(file_in, file_out)    
    mouths = load(file_in);
    mouths = mouths.mouths;
    
    cell_size = 10;
    
    hogs = [];
    
    for i = 1:size(mouths, 1)
        m = squeeze(mouths(i, :, :, :));
        hog = vl_hog(single(m), cell_size);
        hogs(i, :, :, :) = hog;
    end

    save(file_out, 'hogs');
end
