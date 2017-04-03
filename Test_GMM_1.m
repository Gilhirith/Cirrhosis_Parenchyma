function Test_GMM_1()

global n_class;
global img_root_dir;
global class;
global scale_size;
global scale;
global test_samples;

test_cnt = 0;
for i = 4 : 4
    for j = 6 : 20
        if exist([img_root_dir, 'raw_all/cut_', class(i).name, '_', num2str(j), '.jpg'])
            img = im2double(imread([img_root_dir, 'raw_all/cut_', class(i).name, '_', num2str(j), '.jpg']));
            test_cnt = test_cnt + 1;
            if length(size(img)) > 2
                img = rgb2gray(img);
            end
            [ht, wd] = size(img);
            min_y = Mark_Upper_Bound(img);
            Get_Test_Sample_Patch(img, min_y);
            novelty_map = Get_Novelty_Map(img);
            test_samples(test_cnt).novelty_map = novelty_map;
            test_samples(test_cnt).class = i;
            test_samples(test_cnt).wd = wd;
            test_samples(test_cnt).ht = ht - min_y;
            test_samples(test_cnt).sum = sum(novelty_map(:)) / (wd * (ht - min_y));
%             sum(novelty_map(:)) / (wd * (ht - min_y))
            novelty_map = novelty_map - min(novelty_map(:));
            
            peak_res_rw = [];
            peak_cnt_rw = 0;
            for rw = 1 : ht
                [pks, locs] = findpeaks(novelty_map(rw, :));
                peak_res_rw(peak_cnt_rw + 1 : peak_cnt_rw + length(locs), 1) = rw;
                peak_res_rw(peak_cnt_rw + 1 : peak_cnt_rw + length(locs), 2) = locs;
                peak_cnt_rw = peak_cnt_rw + length(locs);
            end
            
            peak_res_co = [];
            peak_cnt_co = 0;
            for co = 1 : wd
                [pks, locs] = findpeaks(novelty_map(:, co));
                peak_res_co(peak_cnt_co + 1 : peak_cnt_co + length(locs), 1) = locs;
                peak_res_co(peak_cnt_co + 1 : peak_cnt_co + length(locs), 2) = co;
                peak_cnt_co = peak_cnt_co + length(locs);
            end
            
            peak_res = intersect(peak_res_co, peak_res_rw, 'rows');
            10000 * size(peak_res, 1) / (wd * (ht - min_y))
            test_samples(test_cnt).num = 10000 * size(peak_res, 1) / (wd * (ht - min_y));
            test_samples(test_cnt).peak_res = peak_res;
            %novelty_map(find(novelty_map<3)) = 0;
%             img_tmp = im2double(imread([img_root_dir, 'raw_all/cut_', class(i).name, '_', num2str(j), '.jpg']));

%             novelty_map(find(novelty_map>25)) = 0;
%             novelty_map(find(img<0.05)) = 0;
            novelty_map = novelty_map / max(novelty_map(:));
            
            clear img_col;
            img_col = zeros(ht, wd);
                        
%             img_col(:, :, 1) = img_tmp;
%             img_col(:, :, 2) = img_tmp;
%             img_col(:, :, 3) = img_tmp;
            img_col(:, :, 1) = img;
            img_col(:, :, 2) = img;
            img_col(:, :, 3) = img;
            img_col(:, :, 1) = img_col(:, :, 1) + novelty_map;
            img_col = img_col / max(img_col(:));
            figure, imshow(img_col);
            hold on;
            
            plot(peak_res(:, 2), peak_res(:, 1), 'r.');
            test_samples(test_cnt).img_novelty = img_col;
            imwrite(img_col, [img_root_dir, 'raw_all/res_', class(i).name, '_', num2str(j), '.jpg']);
            hold off;
            close all;
%             figure, imshow(novelty_map / max(novelty_map(:)));
        end
    end
end