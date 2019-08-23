from own_package.cross_validation import run_skf
from own_package.models.models import create_hparams
from own_package.others import create_results_directory
from own_package.spline_analysis import plot_predicted_splines
import openpyxl

hparams = create_hparams(shared_layers=[50,50], ts_layers=[5,5], cs_layers=[5,5], epochs=5000,reg_l1=0.05, reg_l2=0.,
                         activation='relu', batch_size=16, verbose=0)

write_dir = create_results_directory('./results/skf',
                                     folders=['plots', 'models', 'learning rate plots'],
                                     excels=['skf_results'])

run_skf('conv1', 'ann', 'skf', hparams, norm_mask=None, labels_norm=False,
        loader_file='./excel/Data_loader_spline_full.xlsx',
        skf_file=write_dir + '/skf_results.xlsx',
        skf_sheet=None,
        k_folds=10, k_shuffle=True, save_model=True, save_model_name=None, save_model_dir=write_dir + '/models/',
        plot_name=write_dir + '/learning rate plots/plot')

plot_predicted_splines(write_dir, fn=7)
