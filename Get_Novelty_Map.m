function novelty_map = Get_Novelty_Map(img)

global scale;
global n_scale;
global n_gmm;
global scale_size;

[ht, wd] = size(img);
novelty_map = ones(ht, wd);

for sca = 1 : n_scale
    test(sca).novelty_map = ones(ht, wd);
    scale(sca).tp_test_prob = zeros(n_gmm, scale(sca).test_cnt);
    scale(sca).test_prob = zeros(1, scale(sca).test_cnt);
    
    scale(sca).test_sample = scale(sca).test_sample - repmat(scale(sca).mean_sample, scale(sca).test_cnt, 1);
    scale(sca).test_sample_pca = scale(sca).test_sample * scale(sca).coeff_pca;
    
%     scale(sca).test_sample_pca = PCA_Patch(scale(sca).test_sample);
    scale(sca).test_sample_post = posterior(scale(sca).gmm, scale(sca).test_sample_pca);
    for j = 1 : n_gmm
        scale(sca).tp_test_prob(j, :) = scale(sca).test_sample_post(:, j) .* mvnpdf(scale(sca).test_sample_pca(:, :), scale(sca).gmm.mu(j, :), scale(sca).gmm.Sigma(:, :, j));
    end
    scale(sca).test_prob = -log(sum(scale(sca).tp_test_prob, 1));
    for i = 1 : scale(sca).test_cnt
%         scale(sca).test_sample_pos(i, 1)
%         scale(sca).test_sample_pos(i, 2)
%         ht
%         wd
        test(sca).novelty_map(fix(scale(sca).test_sample_pos(i, 1)) : fix(scale(sca).test_sample_pos(i, 1) + scale_size(sca) - 1), fix(scale(sca).test_sample_pos(i, 2)) : fix(scale(sca).test_sample_pos(i, 2) + scale_size(sca) - 1)) = max(repmat(scale(sca).test_prob(i), scale_size(sca), scale_size(sca)), test(sca).novelty_map(fix(scale(sca).test_sample_pos(i, 1)) : fix(scale(sca).test_sample_pos(i, 1) + scale_size(sca) - 1), fix(scale(sca).test_sample_pos(i, 2)) : fix(scale(sca).test_sample_pos(i, 2) + scale_size(sca) - 1)));
    end
    neg_idx = find(test(sca).novelty_map < 0);
    test(sca).novelty_map(neg_idx) = 0;
    novelty_map = novelty_map .* test(sca).novelty_map;
end

novelty_map = novelty_map .^ (1.0 / n_scale);

end