import numpy as np
from scipy.stats import kstest, genextreme


def estimate_lipschitz_constant(inputs, outputs, Ns=500,
                                ks_p_tol=5e-2, train_ratio=0.8, ord=np.inf,
                                verbose=False):
    assert outputs.ndim == 1, "Only works for 1-dim outputs"
    if np.allclose(outputs, 0, atol=1e-5):
        return 0
    
    inputs = inputs.reshape(-1, 2, inputs.shape[-1])
    outputs = outputs.reshape(-1, 2)
    sjs = np.abs(outputs[:, 1] - outputs[:, 0]) / np.linalg.norm(
        inputs[:, 1] - inputs[:, 0], axis=-1, ord=ord)
    np.random.shuffle(sjs)
    sjs = sjs.reshape(Ns, -1).max(axis=-1)

    train_len = int(len(sjs)*train_ratio)
    sjs_train, sjs_val = sjs[:train_len], sjs[train_len:]
    # fit GEV to {sj} to obtain γˆ and standard error ξ
    shape, loc, scale = genextreme.fit(sjs_train)
    dist = genextreme(shape, loc, scale)
    mx = loc + scale / shape
    # convention is -1/c so we check for <0
    if shape < 0:
        return None

    # validate fit using KS test with significance level 0.05
    res = kstest(sjs_val, dist.cdf)
    if res.pvalue < ks_p_tol:
        if verbose:
            print("Unsuccesful: {:.5} is lower than {}".format(
                res.pvalue, ks_p_tol))
    else:
        if verbose:
            print("Succesful: {:.5} is higher than {}".format(
                res.pvalue, ks_p_tol))
        return mx
    

def evt_gev(samples, Ns=500, ks_p_tol=5e-2, train_ratio=0.8, verbose=False):
    assert samples.ndim == 1, "Only works for 1-dim outputs"
    if np.allclose(samples, 0, atol=1e-5):
        return 0
    samples = samples.copy()
    np.random.shuffle(samples)
    sjs = np.array(samples.reshape(Ns, -1).max(axis=-1))
    train_len = int(len(sjs)*train_ratio)
    sjs_train, sjs_val = sjs[:train_len], sjs[train_len:]
    # fit reverse GEV to {sj} to obtain γˆ and standard error ξ
    shape, loc, scale = genextreme.fit(sjs_train)
    dist = genextreme(shape, loc, scale)
    mx = loc + scale / shape
    # convention is -1/c so we check for <0
    if shape < 0:
        return None
        
    # validate fit using KS test with significance level 0.05
    res = kstest(sjs_val, dist.cdf)
    if res.pvalue < ks_p_tol:
        if verbose:
            print("Unsuccesful: {:.5} is lower than {}".format(
                res.pvalue, ks_p_tol))
    else:
        if verbose:
            print("Succesful: {:.5} is higher than {}".format(
                res.pvalue, ks_p_tol))
        return mx 