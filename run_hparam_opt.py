from own_package.hparam_opt import hparam_opt



hparam_opt('cs', 'hul', norm_mask=[0,1], labels_norm=False, loader_file='./excel/Data_loader_lp4_caa_150219.xlsx',
           total_run=30, instance_per_run=1, hparam_file='./excel/hparams_opt_no_norm.xlsx',
           plot_dir='./Plots/hparam')