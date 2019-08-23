import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from own_package.others import create_results_directory


class Expt():
    """
    Experiment class object that contains all the necessary information and methods to run a baseline experiment for the
    strain sensor project.

    Needs to have/ do:
    1. Ability to randomly generate a selected set of synthesis features in a form of a ndarray of shape
    (no. of examples x no. of features).
    2. Ability to store different methods that are functions that map synthesis features to a continuous real valued
    function z = f(y : x) where z = resistance, y = strain and has domain [0, e(x)] where e is the end point of the
    stain curve which are also dependent on x, x = the set of synthesis features
    3. Ability to create pairs of labelled examples using the above two pointers.
    """

    def __init__(self, results_dir, feature_mode, func_mode, numel, res, plot_mode=False):
        self.write_dir = create_results_directory(results_directory=results_dir, folders=['plots'], excels=['expt'])
        self.numel = numel
        self.res = res
        if feature_mode == 1:
            self.features = self.feature_1()
            feature_headers = np.array(['a', 'b', 'c', 'e'])
        elif feature_mode == 2:
            self.features = self.feature_2()
            feature_headers = np.array(['a', 'b', 'c', 'e'])
        elif feature_mode == 3:
            self.features = self.feature_3()
            feature_headers = np.array(['a', 'b', 'c', 'e'])
        else:
            raise KeyError('feature_mode {} selected does not exist'.format(feature_mode))
        if func_mode == 1:
            func = self.func_1
        elif func_mode == 2:
            func = self.func_2
        elif func_mode == 3:
            func = self.func_3
        else:
            raise KeyError('func_mode {} selected does not exist'.format(func_mode))

        # Generating labels
        self.labels = [func(*item) for item in self.features.tolist()]

        # Writing to excel
        pd_writer = pd.ExcelWriter(self.write_dir + '/expt.xlsx', engine='openpyxl')

        exp_number = np.array(range(self.numel)) + 1  # Index to label Exp 1, 2, 3, ...
        y_number = np.array(range(self.res)) + 1
        labels = [np.concatenate((np.array(item[0]).reshape(-1), item[1])) for item in self.labels]
        summary = np.concatenate((self.features, np.array(labels)), axis=1)
        df_write = pd.DataFrame(summary, index=exp_number,
                                columns=np.concatenate((feature_headers, np.array('e').reshape(-1), y_number)))
        df_write.to_excel(pd_writer)
        pd_writer.save()
        pd_writer.close()

        # Plotting
        if plot_mode:
            for idx, (f, l) in enumerate(zip(self.features.tolist(), self.labels)):
                self.plot(*(l + (idx,) + tuple(f)))

    def feature_1(self):
        """
        Most basic feature set

        Contains a,b,c,e = linear term, poly 4 term, time delay, end point

        :return: ndarray (numel x 4 features) with ending being the last column
        """
        a = np.random.uniform(low=0, high=2, size=(self.numel, 1))
        b = np.random.uniform(low=-10, high=-4, size=(self.numel, 1))
        b = 10 ** b
        c = np.random.uniform(low=0, high=300, size=(self.numel, 1))
        e = c * np.random.uniform(low=1.5, high=3, size=(self.numel, 1))
        return np.concatenate((a, b, c, e), axis=1)

    def feature_2(self):
        """
        Most basic feature set

        Contains a,b,c,e = linear term, poly 4 term, time delay, end point

        :return: ndarray (numel x 4 features) with ending being the last column
        """
        a = np.random.uniform(low=0, high=2, size=(self.numel, 1))
        b = np.random.uniform(low=0, high=2, size=(self.numel, 1))
        c = np.random.uniform(low=0, high=300, size=(self.numel, 1))
        e = c * np.random.uniform(low=1.5, high=3, size=(self.numel, 1))
        return np.concatenate((a, b, c, e), axis=1)

    def feature_3(self):
        """
        Most basic feature set

        Contains a,b,c,e = linear term, poly 4 term, time delay, end point

        :return: ndarray (numel x 4 features) with ending being the last column
        """
        a = np.random.uniform(low=0, high=2, size=(self.numel, 1))
        b = np.random.uniform(low=-4, high=-2, size=(self.numel, 1))
        c = np.random.uniform(low=0, high=300, size=(self.numel, 1))
        e = c * np.random.uniform(low=1.5, high=3, size=(self.numel, 1))
        return np.concatenate((a, b, c, e), axis=1)

    def func_1(self, a, b, c, e):
        """
        Mapping of a,b,c,e to the R vs strain curve
        :return: tuple of (ending point, ndarray of y values)
        """
        x = np.linspace(0, e, num=self.res)
        y = a * (x - c) + b * (x - c) ** 4
        temp = x - c
        temp[temp > 0] = -np.inf
        arg_c = np.argmax(temp)
        y[0:arg_c + 1] = 0
        return e, y

    def func_2(self, a, b, c, e):
        """
        Mapping of a,b,c,e to the R vs strain curve
        :return: tuple of (ending point, ndarray of y values)
        """
        x = np.linspace(0, e, num=self.res)
        y = a * (x - c) + b * (x - c)
        temp = x - c
        temp[temp > 0] = -np.inf
        arg_c = np.argmax(temp)
        y[0:arg_c + 1] = 0
        return e, y

    def func_3(self, a, b, c, e):
        """
        Mapping of a,b,c,e to the R vs strain curve
        :return: tuple of (ending point, ndarray of y values)
        """
        x = np.linspace(0, e, num=self.res)
        y = a * (x - c) + (10**b) * (x - c) ** 2
        temp = x - c
        temp[temp > 0] = -np.inf
        arg_c = np.argmax(temp)
        y[0:arg_c + 1] = 0
        return e, y

    def plot(self, e, y, idx, *args):
        x = np.linspace(0, e, num=self.res)
        plt.plot(x, y, c='b', label='Actual Curve')
        plt.plot([], [], label='Features = {}'.format(args))
        plt.scatter(x, y, c='b', marker='+')
        plt.legend(loc='upper left')
        plt.title('Expt. ' + str(idx + 1))
        plt.savefig(self.write_dir + '/plots' + '/Expt ' + str(idx + 1) + '.png', bbox_inches='tight')
        plt.close()


Expt('./exp_gen', 3, 3, 500, 20)
