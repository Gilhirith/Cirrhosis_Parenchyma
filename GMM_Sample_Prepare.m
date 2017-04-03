function GMM_Sample_Prepare(sca)

global class;
global scale_size;
global sample_cnt;
global scale;
global img_root_dir;

for cls = 1 : 3
    for i = 1 : 400
        if exist([img_root_dir, 'sample_', num2str(scale_size(sca)), '/pos/sample_', num2str(scale_size(sca)), '_', class(cls).name, '_', num2str(i), '.jpg'])
            img = im2double(imread([img_root_dir, 'sample_', num2str(scale_size(sca)), '/pos/sample_', num2str(scale_size(sca)), '_', class(cls).name, '_', num2str(i), '.jpg']));
            sample_cnt = sample_cnt + 1;
            scale(sca).sample(sample_cnt, :) = img(:);
            img = fliplr(img);
            sample_cnt = sample_cnt + 1;
            scale(sca).sample(sample_cnt, :) = img(:);
        end
    end
end

end