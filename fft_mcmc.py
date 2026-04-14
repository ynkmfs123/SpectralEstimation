import numpy as np
from scipy.stats import norm


def fft(x, dt, window=None):
    n = len(x)
    if window is None:
        window = np.ones(n)

    x = x - np.mean(x)
    x = x * window
    spectrum = np.fft.rfft(x)
    power = (np.abs(spectrum) ** 2) / (n * np.mean(window ** 2))
    freq = np.fft.rfftfreq(n, d=dt)
    return freq[1:], power[1:]


def mcmc(f, psd, n_iter, mu, prior_sig, prop_sig, burn=1000, thin=2):
    def model(theta):
        return np.exp(theta[0]) * f ** theta[1] + np.exp(theta[2])

    def logpost(theta):
        if theta[1] >= 0:
            return -np.inf
        log_prior = norm.logpdf(theta, mu, prior_sig).sum()
        s = model(theta)
        log_likelihood = np.sum(-np.log(s) - psd / s)
        return log_prior + log_likelihood

    rng = np.random.default_rng()
    theta = np.array(mu, dtype=float)
    chain = np.empty((n_iter, len(theta)))
    logp = logpost(theta)

    for i in range(n_iter):
        proposal = theta + rng.normal(0, prop_sig)
        proposal_logp = logpost(proposal)

        if np.log(rng.random()) < proposal_logp - logp:
            theta = proposal
            logp = proposal_logp

        chain[i] = theta

    post = chain[burn::thin]
    theta_mean = post.mean(axis=0)
    log_s_fit = np.log(model(theta_mean))

    return log_s_fit, theta_mean, post, chain