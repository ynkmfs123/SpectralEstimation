import numpy as np
import pycwt as wavelet
from scipy.stats import norm
from tqdm import tqdm


def cwt(x, dt, mother, dj, s0, j):
    wave_out, _, freqs, coi, _, _ = wavelet.cwt(x, dt, dj, s0, j, mother)
    power = np.abs(wave_out) ** 2
    periods = 1.0 / freqs
    return power, freqs, coi, periods


def mcmc(
    y,
    freqs,
    n_iter,
    burn_frac,
    thin,
    seed,
    mu_anchor,
    sd_anchor,
    nu_half_t,
    tau_loga,
    tau_alpha,
    tau_logc,
    prop_sig_loga,
    prop_sig_alpha,
    prop_sig_logc,
    prop_sig_logsig,
    show_progress=True,
    progress_desc="MCMC",
):
    def loglike_t(th_t, y_t, logf):
        if th_t[1] >= 0:
            return -np.inf
        m = np.exp(th_t[0]) * np.exp(th_t[1] * logf) + np.exp(th_t[2])
        m = np.clip(m, 1e-300, np.inf)
        return np.sum(-np.log(m) - y_t / m)

    def rw2_logprior_full(series, sigma):
        if series.size < 3:
            return 0.0
        d2 = series[2:] - 2 * series[1:-1] + series[:-2]
        n = d2.size
        return -0.5 * np.sum((d2 / sigma) ** 2) - n * np.log(sigma + 1e-30)

    def rw2_prior_delta_single_component(old_series, new_val, t_idx, sigma):
        t_len = old_series.shape[0]
        if t_len < 3:
            return 0.0

        def d2(series, i):
            return series[i] - 2 * series[i - 1] + series[i - 2]

        idxs = [i for i in (t_idx, t_idx + 1, t_idx + 2) if 2 <= i <= t_len - 1]
        if not idxs:
            return 0.0

        old_sum = sum((d2(old_series, i) / sigma) ** 2 for i in idxs)
        new_series = old_series.copy()
        new_series[t_idx] = new_val
        new_sum = sum((d2(new_series, i) / sigma) ** 2 for i in idxs)
        return -0.5 * (new_sum - old_sum)

    def half_t_logprior_sigma(sigma, nu, tau):
        if sigma <= 0:
            return -np.inf
        return -0.5 * (nu + 1.0) * np.log(1.0 + (sigma / tau) ** 2 / nu) - np.log(tau)

    def logprior_delta_theta(theta_mat, t_idx, prop_th, sigmas, mu_anchor, sd_anchor):
        delta_lp = 0.0
        if t_idx in (0, 1):
            old = theta_mat[:, t_idx]
            delta_lp += (
                norm.logpdf(prop_th, loc=mu_anchor, scale=sd_anchor).sum()
                - norm.logpdf(old, loc=mu_anchor, scale=sd_anchor).sum()
            )
        delta_lp += rw2_prior_delta_single_component(theta_mat[0, :], prop_th[0], t_idx, sigmas[0])
        delta_lp += rw2_prior_delta_single_component(theta_mat[1, :], prop_th[1], t_idx, sigmas[1])
        delta_lp += rw2_prior_delta_single_component(theta_mat[2, :], prop_th[2], t_idx, sigmas[2])
        return delta_lp

    _, t_len = y.shape
    logf = np.log(freqs)
    rng = np.random.default_rng(seed)

    theta = np.tile(mu_anchor[:, None], (1, t_len)).astype(float)
    sigmas = np.array([tau_loga, tau_alpha, tau_logc], dtype=float)
    cur_ll = np.array([loglike_t(theta[:, i], y[:, i], logf) for i in range(t_len)])

    burn = int(n_iter * burn_frac)
    keep_count = 0
    theta_sum = np.zeros_like(theta)
    sigmas_sum = np.zeros_like(sigmas)

    it_range = range(n_iter)
    if show_progress:
        it_range = tqdm(it_range, desc=progress_desc, ncols=90)

    for it in it_range:
        for i in range(t_len):
            prop = theta[:, i].copy()
            prop[0] += rng.normal(0, prop_sig_loga)
            if prop[1] < 0:
                dll = loglike_t(prop, y[:, i], logf) - cur_ll[i]
                dlp = logprior_delta_theta(theta, i, prop, sigmas, mu_anchor, sd_anchor)
                if np.log(rng.random()) < dll + dlp:
                    theta[:, i] = prop
                    cur_ll[i] += dll

            prop = theta[:, i].copy()
            prop[1] += rng.normal(0, prop_sig_alpha)
            if prop[1] < 0:
                dll = loglike_t(prop, y[:, i], logf) - cur_ll[i]
                dlp = logprior_delta_theta(theta, i, prop, sigmas, mu_anchor, sd_anchor)
                if np.log(rng.random()) < dll + dlp:
                    theta[:, i] = prop
                    cur_ll[i] += dll

            prop = theta[:, i].copy()
            prop[2] += rng.normal(0, prop_sig_logc)
            if prop[1] < 0:
                dll = loglike_t(prop, y[:, i], logf) - cur_ll[i]
                dlp = logprior_delta_theta(theta, i, prop, sigmas, mu_anchor, sd_anchor)
                if np.log(rng.random()) < dll + dlp:
                    theta[:, i] = prop
                    cur_ll[i] += dll

        for j, tau in enumerate([tau_loga, tau_alpha, tau_logc]):
            ls_prop = np.log(sigmas[j]) + rng.normal(0, prop_sig_logsig)
            sig_prop = np.exp(ls_prop)
            lp_rw2_prop = rw2_logprior_full(theta[j, :], sig_prop)
            lp_rw2_curr = rw2_logprior_full(theta[j, :], sigmas[j])
            lp_sig_prop = half_t_logprior_sigma(sig_prop, nu_half_t, tau)
            lp_sig_curr = half_t_logprior_sigma(sigmas[j], nu_half_t, tau)
            lp_prop_total = lp_rw2_prop + lp_sig_prop + ls_prop
            lp_curr_total = lp_rw2_curr + lp_sig_curr + np.log(sigmas[j])
            if np.log(rng.random()) < (lp_prop_total - lp_curr_total):
                sigmas[j] = sig_prop

        if it >= burn and ((it - burn) % thin == 0):
            theta_sum += theta
            sigmas_sum += sigmas
            keep_count += 1

    theta_mean = (theta_sum / max(keep_count, 1)).T
    sigmas_mean = sigmas_sum / max(keep_count, 1)

    loga_mean = theta_mean[:, 0]
    alpha_mean = theta_mean[:, 1]
    logc_mean = theta_mean[:, 2]
    bg_spectra = (
        np.exp(loga_mean)[None, :] * (freqs[:, None] ** alpha_mean[None, :])
        + np.exp(logc_mean)[None, :]
    )

    return theta_mean, sigmas_mean, bg_spectra
