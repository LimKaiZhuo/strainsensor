from own_package.hparam_opt import hparam_opt
from own_package.others import create_results_directory

write_dir = create_results_directory('./results/hparams_opt',
                                     folders=['plots', 'models', 'learning rate plots'],
                                     excels=['skf_results', 'hparam_results'])

hparam_opt(model_mode='conv1', loss_mode='ann', norm_mask=[0,0,0,1,1,1], labels_norm=False,
           loader_file='./excel/Data_loader_spline_full_onehot_R2.xlsx',
           total_run=300, instance_per_run=1, write_dir=write_dir,
           plot_dir=None)