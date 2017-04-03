function Get_Test_Sample(img, min_y)

global scale_size;
global n_scale;
global scale;
global step_size;

[ht, wd] = size(img);

for sca = 1 : n_scale;
    scale(sca).test_cnt = 0;
    scale(sca).test_sample = [];
    scale(sca).test_sample_pos = [];
    for i = fix(min_y) : step_size : ht - scale_size(sca) + 1
        for j = 1 : step_size : wd - scale_size(sca) + 1
            tp_img = img(i : i + scale_size(sca) - 1, j : j + scale_size(sca) - 1);
            scale(sca).test_cnt = scale(sca).test_cnt + 1;
            scale(sca).test_sample(scale(sca).test_cnt, :) = tp_img(:);
            scale(sca).test_sample_pos(scale(sca).test_cnt, :) = [i, j];
        end
    end
end


end