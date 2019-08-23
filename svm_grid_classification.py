from own_package.svm_classifier import run_classification
from own_package.others import create_results_directory

write_dir = create_results_directory(results_directory='./results/svm_results')
run_classification(read_dir='./results/grid', write_dir=write_dir)



