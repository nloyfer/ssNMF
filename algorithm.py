import numpy as np
from scipy import optimize

#############################################################
#                                                           #
#             Deconvolution methods                         #
#                                                           #
#############################################################


def calc_RMSE(A, Y, X):
    return np.sqrt(np.power(np.matmul(A, Y) - X, 2).sum())


def run_nnls(A, X, beta):
    nr_samples = X.shape[1]
    nr_ref_samp = A.shape[1]

    # init Y to nan
    Y = np.empty((nr_ref_samp, nr_samples))
    Y[:] = np.nan

    # append sqrt(beta) row to A:
    Ar = np.vstack([A, np.repeat(np.sqrt(beta), nr_ref_samp)])
    # append zero row to A: # TODO: was this Tommy's intention?
    Xr = np.vstack([X, np.repeat(0, nr_samples)])

    # argmin_Y ||AY-X||
    for i in range(nr_samples):
        mixture, residual = optimize.nnls(Ar, Xr[:, i])
        Y[:, i] = mixture
    Y = Y / Y.sum(axis=0)

    return Y


def run_deconvolution(A, X, fixed, beta, eta, n_iter):
    """
    Run NMF
    :param A: reference atlas, size (nr_feat, nr_ref_samp)
    :param X: Data samples, size (nr_feat, nr_samp)
    :param fixed: which columns of A are fixed, size (nr_ref_samp, )
    :param beta: regularization parameter, float
    :param eta: regularization parameter, float
    :param n_iter: number of iterations, int
    :return: the mixture coefficients
    """
    if eta is None:
        eta = X.max() ** 2

    nr_ref_samp = A.shape[1]
    nr_features = X.shape[0]

    history = []
    if fixed.sum() == len(fixed):
        # All columns are fixed. no NMF performed, only NNLS
        Y = run_nnls(A, X, beta)
        history.append(calc_RMSE(A, Y, X))
        return A, Y, history

    # Otherwise, at least some of the columns are not fixed
    fixed_inds = np.argwhere(fixed).flatten()
    o_inds = np.argwhere(1 - fixed).flatten()

    for it in range(n_iter):
        # argmin_Y ||A*Y-X||
        Y = run_nnls(A, X, beta)

        resid = X - np.matmul(A[:, fixed_inds], Y[fixed_inds, :])

        # regularized A by sqrt(eta)
        Xr = np.vstack([resid.T, np.zeros((nr_ref_samp, nr_features))])
        Yr = np.vstack([Y.T, np.sqrt(eta) * np.eye(nr_ref_samp)])

        # argmin_A' ||Y'A'-X'||
        for j in range(nr_features):
            mixture, residual = optimize.nnls(Yr[:, o_inds], Xr[:, j])
            A[j, o_inds] = mixture

        # renormalize columns in A with max>1
        # A = A / np.where(A.max(axis=0) > 1, A.max(axis=0), 1)
        history.append(calc_RMSE(A, Y, X))
    return A, Y, history
