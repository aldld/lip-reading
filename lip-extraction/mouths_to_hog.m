function [] = mouths_to_hog(data_dir)
    speakers = dir([data_dir 's*']);
    
    for j = 1:numel(speakers)
        folder = speakers(j).name;
        folder_path = [data_dir folder]
    end

end

function [] = folder_helper(dir_in, dir_out)
    aligns = dir([dir_in '*.mat']);
    
    for j = 1:numel(aligns)
        align_name = aligns(j).name;    
        align_path = [dir_in align_name];
        hog_name = char(strsplit(align_name, '.'));
        hog_path = [dir_out hog_name(1) '.mat'];
        tic
        filehelper(align_path, hog_path);
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
