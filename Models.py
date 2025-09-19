
import numpy as np
from scipy.linalg import inv
from scipy.spatial.distance import cdist
from itertools import product
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
import numpy.linalg as LA
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import NMF
from scipy.linalg import svd
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
from sklearn.utils import parallel_backend
from sklearn.linear_model import LinearRegression
import time
import functools

def setDist(dist_X=None, dist_Y=None, dist_X_new=None):
    """
    Define the kernel globally to avoid the repeating calculation.

    Parameters
    ----------
    dist_X: array
        Kernel calculated from X.
    dist_Y: array
        Kernel calculated from Y.
    dist_X_new: array
        Kernel calculated from X_new.

    Returns
    -------
    None
    """
    global loaddist
    if (dist_X is None)|(dist_Y is None)|(dist_X_new is None):
        loaddist = False
    else:
        global distance_X
        global distance_Y
        global distance_X_new

        loaddist = True
        distance_X = dist_X
        distance_Y = dist_Y
        distance_X_new = dist_X_new

def runModels(Y,X,X_new,method_option,par=None):
    """
    Function for controlling the prediction during the Cross Validation(CV) or Nested CV.

    Parameters
    ----------
    Y: array
        ADR data.
    X: array
        Feature data.
    X_new: array
        Feature data for new drug.
    method_option: str
        String of method chosen to make prediction.
    par: list
        List of hyperparameter combinations.

    Returns
    -------
    y_new: array 
        Predicted ADR profiles for new drugs.
    """
    if method_option == "SKR":
        lmd = par[0]
        c = par[1]
        sigma_X = par[2]
        sigma_Y = par[3]
        model = SmoothedKR(lmd=lmd,c=c,sigma_X=sigma_X,sigma_Y=sigma_Y)
        model.fit(X=X,Y=Y)
        y_new = model.predict(X_new=X_new)
        return y_new
    elif method_option == "KR":
        lmd = par[0]
        sigma_X = par[1]
        model = KR(lmd=lmd,sigma_X=sigma_X)
        model.fit(X=X,Y=Y)
        y_new = model.predict(X_new=X_new)
        return y_new
    elif method_option == "KRR":
        lmd = par[0]
        sigma_X = par[1]
        model = KRR(lmd=lmd,sigma_X=sigma_X)
        model.fit(X=X,Y=Y)
        y_new = model.predict(X_new=X_new)
        return y_new
    elif method_option == "Naive":
        n,_ = X_new.shape
        model = Naive()
        model.fit(Y=Y)
        y_new = model.predict(n=n)
        return y_new
    elif method_option == "LNSM_RLN":
        alpha = par[0]
        lmd = par[1]
        model = LNSM(alpha=alpha, distance_option="RLN", lmd=lmd)
        model.fit(X=X,Y=Y)
        y_new = model.predict(X_new=X_new)
        return y_new
    elif method_option == "LNSM_jaccard":
        alpha = par[0]
        model = LNSM(alpha=alpha, distance_option="jaccard")
        model.fit(X=X,Y=Y)
        y_new = model.predict(X_new=X_new)
        return y_new
    elif method_option == "VKR":
        sigma = par[0]
        lmd = par[1]
        k = par[2]
        model = VKR(sigma=sigma, lmd=lmd, k=k)
        model.fit(X=X,Y=Y)
        y_new = model.predict(X_new=X_new)
        return y_new
    elif method_option == "SVM":
        return SVM(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "RF":
        k = par[0]
        model = RandomForestRegressor(n_estimators=k, random_state=1949)
        model.fit(X, Y)
        y_new = model.predict(X_new)
        return y_new
    elif method_option == "BRF":
        k = par[0]
        model = BRF(k=k)
        model.fit(X, Y)
        y_new = model.predict(X_new)
        return y_new
    elif method_option == "TNMF":
        return TNMF(Y=Y,X=X,X_new=X_new,par=par)
    else:
        raise ValueError(f"{method_option} is not one of the models.")
    

def loadHyperpar(*hyperpars, method_option):
    """
    Load hyperparameters for methods. Check if the input fits the standard of each method. Generate the hyperparameter combinations according to the range of each hyperparameter.

    Parameters
    ----------
    *hyperpars: list
        List of arrays. Each array represent the range of a hyperparameter.
    method_option: str
        String of method chosen to make prediction.
        

    Returns
    -------
    hyperparList: list
        List of hyperparameter combinations. It is the combinations of arrays in *hyperpars
    """
    if method_option == "SKR":
        n_par = 4
        print(f"The {method_option} requires hyperparameter lambda, c, sigma_X, sigma_Y")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "KR":
        n_par = 2
        print(f"The {method_option} requires hyperparameter lambda, sigma_X")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "KRR":
        n_par = 2
        print(f"The {method_option} requires hyperparameter lambda, sigma_X")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "Naive":
        n_par = 0
        print(f"The {method_option} requires no hyperparameter")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. You do not need to load a hyperparameter list")
    elif method_option == "LNSM_RLN":
        n_par = 2
        print(f"The {method_option} requires hyperparameter alpha and lambda")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "LNSM_jaccard":
        n_par = 1
        print(f"The {method_option} requires hyperparameter alpha")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "VKR":
        n_par = 3
        print(f"The {method_option} requires hyperparameter lambda, sigma_X, k")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "SVM":
        n_par = 2
        print(f"The {method_option} requires hyperparameter c, gamma")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "OCCA":
        n_par = 0
        print(f"The {method_option} requires no hyperparameter")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. You do not need to load a hyperparameter list")
    elif method_option == "SCCA":
        n_par = 2
        print(f"The {method_option} requires hyperparameter alpha")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "RF":
        n_par = 1
        print(f"The {method_option} requires hyperparameter k")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "BRF":
        n_par = 1
        print(f"The {method_option} requires hyperparameter k")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "TNMF":
        n_par = 1
        print(f"The {method_option} requires hyperparameter k")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    else:
        raise ValueError(f"{method_option} is not one of the models.")
    
    hyperparList = list(product(*hyperpars))
    return hyperparList


class SmoothedKR:
    """
    Smoothed Kernel Regression (SKR).
    """
    def __init__(self, lmd, c, sigma_X, sigma_Y):
        """
        Initialize the hyperparmeters

        Parameters
        ----------
        lmd: float
            Value that control the regularization term
        c: float
            Value that control the strengthen of the smoother.
        Sigma_X: float
            Value that control the width of kernel constructed from X.
        Sigma_Y: float
            Value that control the width of kernel constructed from Y.
        """
        self.lmd = lmd
        self.c = c
        self.sigma_X = sigma_X
        self.sigma_Y = sigma_Y

    def normalization(self, K):
        """
        Degree normalization.

        Parameters
        ----------
        K: array
            Matrix requires degree normalization.

        Returns
        ----------
        K_normalized: array
            Normalized matrix.
        
        """
        d1 = K.sum(axis=0) + 10e-8
        d2 = K.sum(axis=1) + 10e-8
        K_normalized = (K.T / np.sqrt(d2)).T / np.sqrt(d1)
        return K_normalized
    
    def smoother(self, Y, K_Y, c, n):
        """
        This function is smoothing Y twice to Y_SS by the smoother. Also the smoothed Y is normalized to the scale of 0 to 1.

        Parameters
        ----------
        Y: array
            Matrix needs to be smoothed.

        Returns
        ----------
        min_max_normalized_matrix: array
            Smoothed and normalized matrix.
        
        """
        Y_SS = ((1-c)*np.diag(np.ones(n))+c/2*self.normalization(K_Y)).dot((1-c)*(np.diag(np.ones(n))+c/2*self.normalization(K_Y))).dot(Y)
        min_val = np.min(Y_SS)
        max_val = np.max(Y_SS)
        min_max_normalized_matrix = (Y_SS - min_val) / (max_val - min_val)
        return min_max_normalized_matrix

    def fit(self, X, Y):
        """
        Fit the model to training set. If the Kernels (K_X, K_Y) was pre-calculated (loaddist == True) then use the pre-calculated values.

        Parameters
        ----------
        Y: array
            Matrix of dependent variable in traing set.
        X: array 
            Matrix of independent variable in traing set.

        Returns
        ----------
        self: self
            Self includes fitted model and necessary variables.
        """
        self.X = X
        n,_ = X.shape # size of known drug
        Lmd = np.diag(np.ones(n)*self.lmd)
        if loaddist == True:
            K_X = (np.exp(-distance_X/self.sigma_X**2))
            K_Y = (np.exp(-distance_Y/self.sigma_Y**2))
        else:
            K_X = (np.exp(-cdist(X, X)**2/self.sigma_X**2))
            K_Y = (np.exp(-cdist(Y, Y)**2/self.sigma_Y**2))

        self.W = inv(K_X.dot(K_X)+Lmd).dot(K_X).dot(self.smoother(Y, K_Y, self.c, n))
        return self
    
    def predict(self, X_new):
        """
        Predict values for new samples. If the Kernels (K_new) was pre-calculated (loaddist == True) then use the pre-calculated values.

        Parameters
        ----------
        X_new: array 
            Matrix of independent variable in test set.

        Returns
        ----------
        y_new: array
            Predicted values.
        """
        if loaddist == True:
            K_new = (np.exp(-distance_X_new/self.sigma_X**2))
        else:
            K_new = (np.exp(-cdist(X_new, self.X)**2/self.sigma_X**2))

        y_new = K_new.dot(self.W)
        return y_new

# class KR:
#     def __init__(self, lmd, sigma_X):
#         self.lmd = lmd
#         self.sigma_X = sigma_X
    
#     def fit(self, X, Y):
#         self.X = X
#         n,_ = X.shape # size of known drug
#         Lmd = np.diag(np.ones(n)*self.lmd)
#         if loaddist == True:
#             K_X = (np.exp(-distance_X/self.sigma_X**2))
#         else:
#             K_X = (np.exp(-cdist(X, X)**2/self.sigma_X**2))
#         self.W = inv(K_X+Lmd).dot(Y)
#         return self

#     def predict(self, X_new):
#         if loaddist == True:
#             K_new = (np.exp(-distance_X_new/self.sigma_X**2))
#         else:
#             K_new = (np.exp(-cdist(X_new, self.X)**2/self.sigma_X**2))
#         y_new = K_new.dot(self.W)
#         return y_new

class KRR:
    """
    Kernel Ridge Regression (KRR)
    """
    def __init__(self, lmd, sigma_X):
        """
        Initialize the hyperparmeters

        Parameters
        ----------
        lmd: float
            Value that control the regularization term
        Sigma_X: float
            Value that control the width of kernel constructed from X.
        """
        self.lmd = lmd
        self.sigma_X = sigma_X
    
    def fit(self, X, Y):
        """
        Fit the model to training set. If the Kernels (K_X) was pre-calculated (loaddist == True) then use the pre-calculated values. 

        Parameters
        ----------
        Y: array
            Matrix of dependent variable in traing set.
        X: array 
            Matrix of independent variable in traing set.

        Returns
        ----------
        self: self
            Self includes fitted model and necessary variables.
        """
        self.X = X
        n,_ = X.shape # size of known drug
        Lmd = np.diag(np.ones(n)*self.lmd)
        if loaddist == True:
            K_X = (np.exp(-distance_X/self.sigma_X**2))
        else:
            K_X = (np.exp(-cdist(X, X)**2/self.sigma_X**2))
        self.W = inv(K_X.dot(K_X)+Lmd).dot(K_X.dot(Y))
        return self

    def predict(self, X_new):
        """
        Predict values for new samples. If the Kernels (K_new) was pre-calculated (loaddist == True) then use the pre-calculated values.

        Parameters
        ----------
        X_new: array 
            Matrix of independent variable in test set.

        Returns
        ----------
        y_new: array
            Predicted values.
        """
        if loaddist == True:
            K_new = (np.exp(-distance_X_new/self.sigma_X**2))
        else:
            K_new = (np.exp(-cdist(X_new, self.X)**2/self.sigma_X**2))
        y_new = K_new.dot(self.W)
        return y_new

class Naive:
    """
    The naive method
    """
    def __init__(self):
        pass

    def fit(self, Y):
        _, self.m = Y.shape
        self.mean_score = (Y.copy()).mean(axis=0)
        return self
    
    def predict(self, n):
        y_new = np.zeros((n, self.m))
        # Set the prediction into mean
        for i in range(self.m):
            y_new[:, i] = self.mean_score[i]
        return y_new

class LNSM():
    """
    Linear Neighbour Similarity Nethod (LNSM)
    """
    def __init__(self, alpha, distance_option, lmd=None, n_neighbors = 200):
        self.alpha = alpha
        self.distance_option = distance_option
        self.lmd = lmd
        self.n_neighbors = n_neighbors
    
    def Jaccard(self, X1, X2):
        W = 1 - cdist(X2, X1, "jaccard")
        return W
    
    def RLN(self, X1, X2, lmd):
        n, _ = X1.shape
        m, _ = X2.shape

        W = np.zeros((m, n))
        clf = Ridge(alpha=lmd)

        N = self.neigh.kneighbors(X2, self.n_neighbors, return_distance=False)
        for i in range(m):
            X_knn_new = X1[N[i], :]
            clf.fit(X_knn_new.T, X2[i, :])
            W[i, N[i]] = clf.coef_
        return W
            
    def fit(self, X, Y):
        Y_0 = (Y.copy())
        self.X = X
        if self.distance_option == "RLN":
            self.neigh = NearestNeighbors(n_neighbors = self.n_neighbors)
            self.neigh.fit(X)
            W = self.RLN(X1=X, X2=X, lmd=self.lmd)
        elif self.distance_option == "jaccard":
            W = self.Jaccard(X1=X, X2=X)

        n, _ = W.shape
        self.Y_1 = (1-self.alpha)*inv(np.diag(np.ones(n)) - self.alpha*W + 1e-8).dot(Y_0)
        return self

    def predict(self, X_new):
        if self.distance_option == "RLN":
            W_new = self.RLN(X1=self.X, X2=X_new, lmd=self.lmd)
        elif self.distance_option == "jaccard":
            W_new = self.Jaccard(X1=self.X, X2=X_new)
        y_new = np.dot(W_new, self.Y_1)
        return y_new


class VKR:
    """
    Kernel Regression on V (VKR)
    """
    def __init__(self, sigma, lmd, k):
        """
        Initialize the hyperparmeters

        Parameters
        ----------
        sigma: float
            Value that control the width of kernel constructed from X.
        lmd: float
            Value that control the regularization term.
        k: int
            Number of component in NMF.
        """
        self.sigma = sigma
        self.lmd = lmd
        self.k = k

    def fit(self, X, Y):
        nmf_model = NMF(n_components=self.k, random_state=1949, max_iter=1000)
        V = nmf_model.fit_transform(Y)
        self.U = nmf_model.components_

        self.krr_model = KRR(lmd=self.lmd,sigma_X=self.sigma)
        self.krr_model.fit(X=X,Y=V)
        return self
    
    def predict(self, X_new):
        Vpreds = self.krr_model.predict(X_new=X_new)
        y_new = Vpreds.dot(self.U)
        return y_new

def SVMloop(idx,Y,X,X_new,par):
    c = par[0]
    g = par[1]
    y_i = np.array(Y[:, idx].copy()).tolist()
    if len(np.unique(y_i)) == 1:
        y_new_SVM[:, idx] = np.unique(y_i)
    else:
        svr = SVR(kernel="rbf", C = c, gamma = g).fit(X, y_i)
        y_new_SVM[:, idx] = svr.predict(X_new)


def SVM(Y,X,X_new,par):
    X_normalized = (X.copy() - X.mean(axis=0)) / (X.std(axis=0) + 10e-8)
    m,_ = X_new.shape
    _,n = Y.shape
    global y_new_SVM
    y_new_SVM = np.zeros((m, n)).astype(float)
    
    with parallel_backend('threading'):
        Parallel(n_jobs=1)(delayed(SVMloop)(idx = i, Y=Y, X=X_normalized, X_new=X_new,par=par) for i in range(n))

    return y_new_SVM


class BRF:
    def __init__(self, k, iter=50, random_state=1949):
        self.k = k
        self.iter = iter
        self.lr = 1/iter
        self.random_state = random_state
    def fit(self, X, Y):
        rf = RandomForestRegressor(n_estimators=self.k, random_state=self.random_state)
        rf.fit(X, Y)
        return self
    def predict(self, X_new):
        y_new = self.rf.predict(X_new)
        for i in range(self.iter-1):
            y_new += self.rf.predict(X_new)
        return self.lr*y_new


# def TNMF(Y,X,X_new,par):
#     k = par[0]
#     # lmd = par[1]
#     n, _ = X_new.shape
#     _, m = Y.shape
#     y_new_0 = np.zeros((n, m))
#     XX = np.vstack((X_new, X))
#     YY = np.vstack((y_new_0, Y))

#     Similarity = 1-cdist(XX, XX)**2
#     WH = np.hstack((YY, XX))

#     nmf_model = NMF(n_components=k, random_state=1949, max_iter=10000)
#     W = nmf_model.fit_transform(WH)
#     H = nmf_model.components_
#     y_new = np.dot(W, H)[0:n, 0:m]
#     # _, _, reconWH = GNMF(bipart_graph=WH.T, component=k, WMK=Similarity, lmd=lmd, max_iter=10000, tolerance=1/10000)
#     # y_new = reconWH.T[0:n, 0:m]
#     return y_new