from own_package.cross_validation import run_skf
from own_package.models.models import create_hparams
from own_package.others import create_results_directory
from own_package.spline_analysis import plot_arcsinh_predicted_splines, plot_cutoff
from own_package.features_labels_setup import load_data_to_fl
import openpyxl


def run_skf_conv1(inputs, plot_spline):
    shared, end, pre, filters, epochs, label_type = inputs
    hparams = create_hparams(shared_layers=[30,30], ts_layers=[5,5], cs_layers=[5,5],
                             shared=shared, end=end, pre=pre, filters=filters, epochs=epochs,
                             reg_l1=0.05, reg_l2=0.,
                             activation='relu', batch_size=16, verbose=0)

    write_dir = create_results_directory('./results/skf',
                                         folders=['plots', 'models', 'learning rate plots'],
                                         excels=['skf_results'])
    fl = load_data_to_fl('./excel/Data_loader_spline_full_onehot_R6e.xlsx',
                         label_type=label_type,
                         normalise_labels=False,
                         norm_mask=[0, 0, 0, 1, 1, 1])
    fl_store = fl.create_kf(k_folds=10, shuffle=True)

    run_skf(model_mode='ann3', loss_mode='ann', fl=fl, fl_store=fl_store, hparams=hparams,
            skf_file=write_dir + '/skf_results.xlsx',
            skf_sheet=None,
            k_folds=10, k_shuffle=True, save_model=True, save_model_name=None, save_model_dir=write_dir + '/models/',
            plot_name=write_dir + '/learning rate plots/plot')
    if plot_spline:
        if label_type=='points':
            plot_arcsinh_predicted_splines(plot_dir='{}/plots'.format(write_dir),
                                           results_excel_dir='{}/skf_results.xlsx'.format(write_dir),
                                           end_excel_dir='./results/combine Round 6/end 6e.xlsx',
                                           sheets=['ann3'], fn=6, numel=100)
        elif label_type=='cutoff':
            plot_cutoff(plot_dir='{}/plots'.format(write_dir),
                        results_excel_dir='{}/skf_results.xlsx'.format(write_dir),
                        sheets=['ann3'], fn=6, numel=3)

    return write_dir


inputs = [0,0,70,0,500, 'points']
write_dir = run_skf_conv1(inputs, plot_spline=True)
#plot_predicted_splines(write_dir='./results/skf10 archsinh', excel_dir='./results/skf10 archsinh/skf_results.xlsx', sheets=['conv1'], fn=6)