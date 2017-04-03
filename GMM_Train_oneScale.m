function GMM_Train_oneScale(sca)

global scale;
global n_gmm;

% perform PCA on samples
scale(sca).sample_pca = PCA_Patch(sca, scale(sca).sample);

% train GMM on the samples after PCA
scale(sca).gmm = fitgmdist(scale(sca).sample_pca, n_gmm, 'Regularize', 0.01);
scale(sca).sample_post = posterior(scale(sca).gmm, scale(sca).sample_pca);
for j = 1 : n_gmm
    scale(sca).tp_prob(j, :) = scale(sca).sample_post(:, j) .* mvnpdf(scale(sca).sample_pca(:, :), scale(sca).gmm.mu(j, :), scale(sca).gmm.Sigma(:, :, j));
end
scale(sca).prob = -log(sum(scale(sca).tp_prob, 1));
scale(sca).v_gm = fitgmdist(scale(sca).prob', 1);
scale(sca).maxv = scale(sca).v_gm.mu;% + 2 * scale(sca).v_gm.Sigma;

end