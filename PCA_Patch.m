function sample_pca = PCA_Patch(sca, sample)

global scale;
global PCA_dim;

[coeff, sample_pca, latent] = pca(sample, 'NumComponents', PCA_dim);
tp_sum = cumsum(latent) ./ sum(latent);
scale(sca).coeff_pca = coeff;
scale(sca).mean_sample = mean(sample, 1);
% [coeff,score,latent,tsquared,explained,mu] = pca(sample, 'NumComponents', PCA_dim);

end