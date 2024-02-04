"""
Hawkes process with exponential kernel.
"""
import numpy as np


def hawkes1(n_samples):
    mu = 0.2
    alpha = [0.8, 0.0]
    beta = [1.0, 20.0]
    arri, loglike, inte, dens, cumu = _sample_and_nll(n_samples, mu, alpha, beta)
    return arri, loglike, inte, dens, cumu


def hawkes2(n_samples):
    mu = 0.2
    alpha = [0.4, 0.4]
    beta = [1.0, 20.0]
    arri, loglike, inte, dens, cumu = _sample_and_nll(n_samples, mu, alpha, beta)
    return arri, loglike, inte, dens, cumu


def _sample_and_nll(n_samples, mu, alpha, beta):

    T = []
    LL = []
    DEN = []
    INT = []
    CUMU = []

    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0

    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step

        l_trg_Int1 += l_trg1 * (1 - np.exp(-beta[0]*step)) / beta[0]
        l_trg_Int2 += l_trg2 * (1 - np.exp(-beta[1]*step)) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2

        if np.random.rand() < l_next/l: #accept
            T.append(x)
            LL.append(np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int)
            INT.append(np.exp(l_next))
            DEN.append(np.exp(np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int))
            CUMU.append(1 - np.exp(-(l_trg_Int1 + l_trg_Int2 + mu_Int)))
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1

            if count == n_samples:
                break

    return np.array(T), np.array(LL), np.array(INT), np.array(DEN), np.array(CUMU)


arrival_times = []
log_likelihood = []
intensity = []
density = []
cumulative = []
for i in range(64):
    arrival_times_, log_likelihood_, intensity_, density_, cumulative_ = hawkes1(1024)
    arrival_times.append(arrival_times_)
    log_likelihood.append(log_likelihood_)
    intensity.append(intensity_)
    density.append(density_)
    cumulative.append(cumulative_)

arrival_times = np.array(arrival_times)
log_likelihood = np.array(log_likelihood)
intensity = np.array(intensity)
density = np.array(density)
cumulative = np.array(cumulative)
print(-np.mean(log_likelihood))
np.savez("/data/synth_ours/hawkes1.npz", arrival_times=arrival_times,
         log_likelihood=log_likelihood, intensity=intensity, density=density, cumulative=cumulative)
