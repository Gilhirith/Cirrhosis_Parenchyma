function GMM_Train()

global n_scale;
global scale;
global sample_cnt;

rng('default');  % For reproducibility

sample_cnt = 0;

for sca = 1 : n_scale
    GMM_Sample_Prepare(sca);
    GMM_Train_oneScale(sca);
end

end