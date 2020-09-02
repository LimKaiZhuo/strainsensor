import numpy as np
from own_package.inverse_design import inverse_design, eval_models


def selector(case, **kwargs):
    if case==1:
        bounds = [[0, 1],
                  [0, 1],
                  [200, 2000],
                  [0, 2]]
        inverse_design(targets=np.array([0.1,0.8,7.5]), bounds=bounds, int_idx=[3], init_guess=None,
                       opt_mode='dummy',
                       model_directory_store=['./results/inverse_design/models/ann invariant',
                                           './results/inverse_design/models/ann NDA',
                                           './results/inverse_design/models/ann NDA - 4',
                                           './results/inverse_design/models/dtr I10',],
                       svm_directory='./results/svm gamma130/models',
                       loader_file='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
                       write_dir='./results/inverse_design',
                       )
    elif case==2:
        eval_models(model_directory_store=['./results/inverse_design/models/ann invariant',
                                           './results/inverse_design/models/ann NDA',
                                           './results/inverse_design/models/ann NDA - 4',
                                           './results/inverse_design/models/dtr I10',], results_dir='./results/inverse_design')

selector(1)