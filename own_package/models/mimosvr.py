# https://github.com/sebastian-montoyaj/MIMO-SVR-Python-
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA


# Celda equivalente al archivo o funcion: scale.m
def scale(X):
    mu = np.amin(X, axis=0)
    sig = np.amax(X, axis=0) - np.amin(X, axis=0)
    X_Normal = (X - mu) / sig

    return X_Normal, mu, sig

# Celda equivalente al archivo o funcion: kernelmatrix.m

# ---------------------------------------------------------------------------------------------------
# KERNELMATRIX
#
# K = kernelmatrix(ker,X,X2,parameter);
#
# Construye un kernel a partir de los datos de entrenamiento y validacion
#
# Entradas/Parametros:
#      ker: {'lin' 'poly' 'rbf'}
#      X: Matriz de datos con las muestras de entrenamiento en filas y las caracteristicas en columnas
#      X2: Matriz de datos con las muestras de validacion en filas y las caracteristicas en columnas
#      parameter: Depende el tipo de kernel a emplear, por lo que puede ser::
#         El ancho del kernel RBF
#         El termino independiente en el kernel lineal
#         El grado del kernel polinomial
#
# Salida:
#      K: kernel matrix
#
# ---------------------------------------------------------------------------------------------------

def kernelmatrix(ker, X, X2, parameter):
    if ker == 'lin':
        if X2 is None:
            K = np.dot(X.T, X / (np.linalg.norm(np.dot(X.T, X)))) + parameter  # Aun me queda la duda
            return K
        else:
            K = np.dot(X.T, X2 / (np.linalg.norm(np.dot(X.T, X2)))) + parameter
            return K

    elif ker == 'poly':
        if X2 is None:
            K = (np.dot(X.T, X) / (np.linalg.norm(np.dot(X.T, X))) + 1) ** parameter  # Aun me queda la duda
            return K
        else:
            K = (np.dot(X.T, X2) / (np.linalg.norm(np.dot(X.T, X2))) + 1) ** parameter
            return K

    elif ker == 'rbf':

        n1sq = np.sum(X ** 2, axis=0)
        n1sq = np.atleast_2d(n1sq)
        n1 = np.size(X, 1)

        if n1 == 1:  # Solo una caracteristica
            N1 = np.size(X, 0)
            N2 = np.size(X2, 0)

            D = np.zeros((N1, N2))

            for i in range(0, N1):
                D[i, :] = (X2 - np.ones((N2, 1)) * X[i, :]).T * (X2 - np.ones((N2, 1)) * X[i, :]).T
        else:
            if X2 is None:
                D = (np.ones((n1, 1)) * n1sq).T + np.ones((n1, 1)) * n1sq - 2 * np.dot(X.T, X)
            else:
                n2sq = np.sum(X2 ** 2, axis=0)
                n2sq = np.atleast_2d(n2sq)
                n2 = np.size(X2, 1)

                D = (np.ones((n2, 1)) * n1sq).T + np.ones((n1, 1)) * n2sq - 2 * np.dot(X.T, X2)

        K = np.exp(-D ** 2 / (2 * (parameter ** 2)))
        return K
    else:
        print("ERROR kernel")



# Celda equivalente al archivo o funcion: msvr.m

# ---------------------------------------------------------------------------------------------------
# SVR Multiples saliddas
# Modelo que tiene m ejemplos de entrenamiento, d dimensiones(caracteristicas) y k salidas a predecir
#
# Entradas:   - x : muestras de entrenamiento (m x d),
#             - y : salidas del sistema (m x k),
#             - ker : tipo de kernel ('lin', 'poly', 'rbf'),
#             - C : parametro de costo (boxConstraint),
#             - par : parametro asociado al tipo de kernel (ver funcion 'kernelmatrix') ,
#             - tol : tolerancia.
#
# Salidas:    - Beta : matriz o conjunto solucion con los parametros de la regresion,
#             - NSV : numero de vectores de soporte,
#             - H : kernel matrix,
#             - i1 : indices de cuales muestras son los vectores de soporte.
# ---------------------------------------------------------------------------------------------------

def msvr(x, y, ker, C, epsi, par, tol):
    n_m = np.size(x, 0)  # Obtengo el numero de muestras
    n_d = np.size(x, 1)  # Obtengo el numero de caracteristicas del problema
    n_k = np.size(y, 1)  # Obtengo el numero de salidas a predecir

    # Construyo la matriz kernel a partir de los muestras dadas
    H = kernelmatrix(ker, x.T, x.T, par)

    # Creo una matriz para ir calculando iterativamente los parametros de la regresion
    Beta = np.zeros((n_m, n_k))

    # E = Error de la prediccion por cada una de las salidas (Dimensiones: m x k)
    E = y - np.dot(H, Beta)
    u = np.sqrt(np.sum(E ** 2, 1))

    # RMSE (Raiz cuadrada del error)
    RMSE = np.sqrt(np.mean(u ** 2))

    # Se obtienen los indices de aquellas muestras cuyo error de prediccion es mayor que epsilon
    i1 = np.where(u >= epsi)[0]

    # Ahora, se establecen los valores iniciales de los alphas (terminos independientes)
    a = 2 * C * (u - epsi) / u

    # Creo un vector L  con dimensiones: 1 x m (L es el margen).
    L = np.zeros(np.size(u))

    # Se modifican solo las muestras que cumplen que u > epsi
    L[i1] = u[i1] ** 2 - 2 * epsi * u[i1] + epsi ** 2

    # Lp es la cantidad a minimizar
    Lp = sum(np.diag(np.dot(np.dot(Beta.T, H), Beta))) / 2 + C * sum(L) / 2

    eta = 1
    k = 1
    hacer = 1
    val = 1

    while hacer:
        Beta_a = Beta
        E_a = E
        u_a = u
        i1_a = i1

        M1 = (H[np.ix_(i1, i1)] + np.diag(1 / a[i1])) + 1e-10 * np.eye(len(a[i1]))

        # recalculo los betas
        sal1 = np.dot(np.linalg.inv(M1), y[i1])

        eta = 1
        Beta = np.zeros(np.shape(Beta))
        Beta[i1, :] = sal1

        # recalculo el error
        E = y - np.dot(H, Beta)

        # RSE
        u = np.sqrt(np.sum(E ** 2, 1))
        i1 = np.where(u >= epsi)[0]

        L = np.zeros(np.size(u))

        L[i1] = u[i1] ** 2 - 2 * epsi * u[i1] + epsi ** 2

        # Se recalcula la funcion de perdida
        Lp = np.append(Lp, sum(np.diag(np.dot(np.dot(Beta.T, H), Beta))) / 2 + C * sum(L) / 2)

        # Ciclo donde se guardaran los alphas y se modificaran los betas
        while (Lp[k] > Lp[k - 1]):
            eta = eta / 10
            i1 = i1_a

            Beta = np.zeros(np.shape(Beta))

            # Los nuevos betas son una combinacion de los betas actuales (o sea, sal1) y
            # los betas de la iteracion anterior (o sea, Beta_a)
            Beta[i1] = eta * sal1 + (1 - eta) * Beta_a[i1, :]

            E = y - np.dot(H, Beta)
            u = np.sqrt(np.sum(E ** 2, 1))
            i1 = np.where(u >= epsi)[0]

            L = np.zeros(np.size(u))
            L[i1] = u[i1] ** 2 - 2 * epsi * u[i1] + epsi ** 2
            Lp[k] = sum(np.diag(np.dot(np.dot(Beta.T, H), Beta))) / 2 + C * sum(L) / 2

            # Criterio de parada Num. 1
            if (eta < 10 ** -16):
                Lp[k] = Lp[k - 1] - 10 ** -15
                Beta = Beta_a
                # ---------------
                u = u_a
                i1 = i1_a
                # ---------------
                hacer = 0

        # Aqui modificamos(actualizamos) los alphas y guardamos los betas.
        a_a = a
        a = 2 * C * (u - epsi) / u

        RMSE = np.append(RMSE, np.sqrt(np.mean(u ** 2)))

        # Criterio de parada Num. 2
        if ((Lp[k - 1] - Lp[k]) / Lp[k - 1] < tol):
            hacer = 0

        k = k + 1

        # Criterio de parada Num. 3 (Algoritmo no converge, por tanto: val = -1)
        if (np.shape(i1)[0] == 0):
            hacer = 0
            Beta = np.zeros(np.shape(Beta))
            val = -1

    NSV = np.shape(i1)[0]

    pred = np.dot(H, Beta)

    return Beta, NSV, H, i1

"""
#### Ejemplo de como se implementaria M-SVR ####

# Aqui se cargan los datos de entrada y salida del sistema
datosEntradas = pd.read_csv('Entradas_Reducidas.csv', usecols=range(3, 71))
archivoSalidas = pd.ExcelFile('Salidas.xlsx')
datosSalidas = pd.read_excel(archivoSalidas, "Sheet1")

# PARAMETROS del sistema
C = 10  # Parametro de regularización (boxConstraint)
paramK = 10  # Parametro para la funcion kernel
tipoK = 'rbf'  # Tipo de kernel a usar en el modelo. Los disponibles son: 'lin', 'poly' y 'rbf'
epsilon = 1  # Parametro que establece el ancho del epsilon tubo o zona de tolerancia (por defecto es 1)
tol = 10 ** -20  # Parametro que necesita el modelo para detener el modelo si el error se vuelve muy bajo

# Establezco el numero de pliegues (folds) con los que validare el sistema
num_Subconjuntos = 10
pliegues = KFold(len(datosEntradas), n_folds=num_Subconjuntos)

# También creo varios arreglos de igual numero de espacios que iteraciones y en donde guardare los errores
ECMTest = np.zeros((num_Subconjuntos, np.shape(datosSalidas)[1]))
iteracion = 0

# Ahora realizare tantas iteraciones como pliegues haya definido, donde:
for train_index, test_index in pliegues:
    # Selecciono los subconjuntos de entrenamiento y validacion para esta iteracion
    X_train = datosEntradas.ix[train_index]
    Y_train = datosSalidas.ix[train_index]
    X_test = datosEntradas.ix[test_index]
    Y_test = datosSalidas.ix[test_index]

    # Se normalizan las X's
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # Se remueve la media de las Y's (solo para el entrenamiento)
    Ntrain = np.shape(X_train)[0]
    mean_Y_train = np.mean(Y_train, axis=0)
    Y_train = Y_train - np.matlib.repmat(mean_Y_train, Ntrain, 1)

    # Entrenamiento del modelo
    Beta, numero_VectoresS, kernel_train, indices_vectoresS = msvr(X_train.values, Y_train.values, tipoK, C, epsilon,
                                                                   paramK, tol)

    # Predicción del modelo
    kernel_test = kernelmatrix(tipoK, X_test.T, X_train.T, paramK)
    Ntest = np.shape(X_test)[0]
    b = np.matlib.repmat(mean_Y_train, Ntest, 1)
    Y_est = np.dot(kernel_test, Beta) + b

    # Medicion de ECM
    ECMTest[iteracion] = (np.sum((Y_est - Y_test.values), axis=0) ** 2) / np.shape(Y_test)[0]


    iteracion = iteracion + 1

# Finalmente, muestro los resultados finales

mean_ECM = np.mean(ECMTest, axis=0)
ic_ECM = np.std(ECMTest, axis=0)
"""
