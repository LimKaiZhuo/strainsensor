from own_package.cross_validation import run_skf
from own_package.models.models import create_hparams
from own_package.others import create_results_directory
from own_package.spline_analysis import plot_predicted_splines
import openpyxl


def run_skf_conv1(inputs):
    shared, end, pre, filters, epochs = inputs
    hparams = create_hparams(shared_layers=[30,30], ts_layers=[5,5], cs_layers=[5,5],
                             shared=shared, end=end, pre=pre, filters=filters, epochs=epochs,
                             reg_l1=0.05, reg_l2=0.,
                             activation='relu', batch_size=16, verbose=0)

    write_dir = create_results_directory('./results/skf',
                                         folders=['plots', 'models', 'learning rate plots'],
                                         excels=['skf_results'])

    run_skf('conv1', 'ann', 'skf', hparams, norm_mask=[0,0,0,1,1,1], labels_norm=False,
            loader_file='./excel/Data_loader_spline_full_onehot_R2.xlsx',
            skf_file=write_dir + '/skf_results.xlsx',
            skf_sheet=None,
            k_folds=10, k_shuffle=True, save_model=True, save_model_name=None, save_model_dir=write_dir + '/models/',
            plot_name=write_dir + '/learning rate plots/plot')

    plot_predicted_splines(write_dir=write_dir, fn=6)


inputs = [88,91,158,2,358]
run_skf_conv1(inputs)

inputs = [155,168,183,3,902]
run_skf_conv1(inputs)

inputs = [144,102,161,6,155]
run_skf_conv1(inputs)

inputs = [132,84,168,1,1038]
run_skf_conv1(inputs)

inputs = [63,117,101,3,701]
run_skf_conv1(inputs)
