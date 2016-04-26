function [] = mouths_to_hog(file_in, file_out)
    mouths = load(file_in);
    mouths = mouths.mouths;
    
    cell_size = 6;
    
    hogs = [];
    
    for i = 1:size(mouths, 1)
        m = squeeze(mouths(i, :, :, :));
        hog = vl_hog(single(m), cell_size);
        hogs(i, :, :, :) = hog;
    end

    save(file_out, 'hogs');
end
