import multiprocessing
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import Interped
from joblib import Parallel, delayed
from numpy.random import uniform as unif
from tqdm.auto import tqdm

from gw_ml_priors.conversions import calc_a2
from gw_ml_priors.regressors.scikit_regressor import ScikitRegressor

NUM_CORES = multiprocessing.cpu_count()


def get_a1_prior(xeff, q, mcmc_n=int(1e5)):
    a1s = np.linspace(0, 1, 500)
    da1 = a1s[1] - a1s[0]
    p_a1 = Parallel(n_jobs=NUM_CORES, verbose=1)(
        delayed(get_p_a1_given_xeff_q)(a1, xeff, q, mcmc_n)
        for a1 in tqdm(a1s, desc="Building a1 cache")
    )
    p_a1 = p_a1 / np.sum(p_a1) / da1
    data = pd.DataFrame(dict(a1=a1s, p_a1=p_a1))
    a1 = data.a1.values
    p_a1 = norm_values(data.p_a1.values, a1)
    min_b, max_b = find_boundary(a1, p_a1)
    return Interped(
        xx=a1, yy=p_a1, minimum=min_b, maximum=max_b, name="a_1", latex_label=r"$a_1$"
    )


def get_p_a1_given_xeff_q(a1, xeff, q, n=int(1e4)):
    cos1, cos2 = unif(-1, 1, n), unif(-1, 1, n)
    a2 = calc_a2(xeff=xeff, q=q, cos1=cos1, cos2=cos2, a1=a1)
    integrand = a2_interpreter_function(a2)
    return np.mean(integrand)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_boundary_idx(x):
    """finds idx where data is non zero (assumes that there wont be gaps)"""
    non_z = np.nonzero(x)[0]
    return non_z[0], non_z[-1]


def norm_values(y, x):
    return y / np.trapz(y, x)


def find_boundary(x, y):
    b1, b2 = find_boundary_idx(y)
    vals = [x[b1], x[b2]]
    start, end = min(vals), max(vals)
    return start, end


def a2_interpreter_function(a2):
    return np.where(((0 < a2) & (a2 < 1)), 1, 0)


def main():
    outdir = "out_p_a1_given_q_xeff"
    os.makedirs(outdir, exist_ok=True)
    q, xeff = 0.5, -0.4
    a1_prior = get_a1_prior(q=q, xeff=xeff)
    plt.plot(a1_prior.xx, a1_prior.yy)
    plt.xlabel("a1")
    plt.ylabel(f"p(a1|q={q},xeff={xeff})")
    plt.savefig(f"{outdir}/p_a1_given_q_xeff.png")

    ## TODO: train ML model to generate p(a1|xeff,q) for different q, xeff


if __name__ == "__main__":
    main()
