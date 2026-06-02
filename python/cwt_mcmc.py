
import numpy as np
import pycwt as wavelet
from tqdm import tqdm
from numba import njit


def cwt(x, dt, mother, dj, s0, j):
    wave_out, _, freqs, coi, _, _ = wavelet.cwt(
        x, dt, dj, s0, j, mother
    )

    power = np.abs(wave_out) ** 2
    periods = 1.0 / freqs

    return power, freqs, coi, periods


# =========================================================
# NUMBA FUNCTIONS
# =========================================================

@njit(fastmath=True)
def loglike_single(loga, alpha, logc, y_col, logf):

    a = np.exp(loga)
    c = np.exp(logc)

    ll = 0.0

    for k in range(logf.shape[0]):

        model = a * np.exp(alpha * logf[k]) + c

        if model < 1e-300:
            model = 1e-300

        ll += -np.log(model) - y_col[k] / model

    return ll


@njit(fastmath=True)
def rw2_prior_full(series, sigma):

    n = series.shape[0]

    if n < 3:
        return 0.0

    s = 0.0

    for i in range(2, n):

        d2 = series[i] - 2.0 * series[i - 1] + series[i - 2]
        s += (d2 / sigma) ** 2

    return -0.5 * s - (n - 2) * np.log(sigma + 1e-30)


@njit(fastmath=True)
def rw2_delta(series, idx, oldv, newv, sigma):

    n = series.shape[0]

    if n < 3:
        return 0.0

    old_sum = 0.0
    new_sum = 0.0

    for i in range(max(2, idx), min(n, idx + 3)):

        d2_old = (
            series[i]
            - 2.0 * series[i - 1]
            + series[i - 2]
        )

        old_sum += (d2_old / sigma) ** 2

    tmp = series[idx]
    series[idx] = newv

    for i in range(max(2, idx), min(n, idx + 3)):

        d2_new = (
            series[i]
            - 2.0 * series[i - 1]
            + series[i - 2]
        )

        new_sum += (d2_new / sigma) ** 2

    series[idx] = tmp

    return -0.5 * (new_sum - old_sum)


@njit(fastmath=True)
def half_t_prior(sigma, nu, tau):

    if sigma <= 0:
        return -1e300

    return (
        -0.5 * (nu + 1.0)
        * np.log(1.0 + (sigma / tau) ** 2 / nu)
        - np.log(tau)
    )


# =========================================================
# MAIN MCMC
# =========================================================

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

    rng = np.random.default_rng(seed)

    n_freq, t_len = y.shape

    logf = np.log(freqs)

    # -----------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------

    theta = np.tile(mu_anchor[:, None], (1, t_len)).astype(np.float64)

    sigmas = np.array(
        [tau_loga, tau_alpha, tau_logc],
        dtype=np.float64
    )

    cur_ll = np.zeros(t_len)

    for i in range(t_len):

        cur_ll[i] = loglike_single(
            theta[0, i],
            theta[1, i],
            theta[2, i],
            y[:, i],
            logf
        )

    # -----------------------------------------------------
    # STORAGE
    # -----------------------------------------------------

    burn = int(n_iter * burn_frac)

    keep_count = 0

    theta_sum = np.zeros_like(theta)
    sigma_sum = np.zeros_like(sigmas)

    # -----------------------------------------------------
    # ITERATION
    # -----------------------------------------------------

    iterator = range(n_iter)

    if show_progress:
        iterator = tqdm(iterator, desc=progress_desc, ncols=90)

    for it in iterator:

        # =================================================
        # UPDATE THETA
        # =================================================

        for i in range(t_len):

            old_loga = theta[0, i]
            old_alpha = theta[1, i]
            old_logc = theta[2, i]

            prop_loga = old_loga + rng.normal(0, prop_sig_loga)
            prop_alpha = old_alpha + rng.normal(0, prop_sig_alpha)
            prop_logc = old_logc + rng.normal(0, prop_sig_logc)

            # alpha must remain negative
            if prop_alpha >= 0:
                continue

            # ---------------------------------------------
            # LIKELIHOOD
            # ---------------------------------------------

            prop_ll = loglike_single(
                prop_loga,
                prop_alpha,
                prop_logc,
                y[:, i],
                logf
            )

            dll = prop_ll - cur_ll[i]

            # ---------------------------------------------
            # PRIOR
            # ---------------------------------------------

            dlp = 0.0

            if i <= 1:

                old_anchor = (
                    -0.5 * np.sum(
                        ((theta[:, i] - mu_anchor) / sd_anchor) ** 2
                    )
                )

                new_anchor = (
                    -0.5 * np.sum(
                        (
                            (
                                np.array([
                                    prop_loga,
                                    prop_alpha,
                                    prop_logc
                                ]) - mu_anchor
                            ) / sd_anchor
                        ) ** 2
                    )
                )

                dlp += new_anchor - old_anchor

            dlp += rw2_delta(
                theta[0, :],
                i,
                old_loga,
                prop_loga,
                sigmas[0]
            )

            dlp += rw2_delta(
                theta[1, :],
                i,
                old_alpha,
                prop_alpha,
                sigmas[1]
            )

            dlp += rw2_delta(
                theta[2, :],
                i,
                old_logc,
                prop_logc,
                sigmas[2]
            )

            # ---------------------------------------------
            # ACCEPT
            # ---------------------------------------------

            if np.log(rng.random()) < (dll + dlp):

                theta[0, i] = prop_loga
                theta[1, i] = prop_alpha
                theta[2, i] = prop_logc

                cur_ll[i] = prop_ll

        # =================================================
        # UPDATE SIGMAS
        # =================================================

        taus = np.array(
            [tau_loga, tau_alpha, tau_logc]
        )

        for j in range(3):

            old_sigma = sigmas[j]

            prop_logs = (
                np.log(old_sigma)
                + rng.normal(0, prop_sig_logsig)
            )

            prop_sigma = np.exp(prop_logs)

            lp_old = (
                rw2_prior_full(theta[j, :], old_sigma)
                + half_t_prior(
                    old_sigma,
                    nu_half_t,
                    taus[j]
                )
                + np.log(old_sigma)
            )

            lp_new = (
                rw2_prior_full(theta[j, :], prop_sigma)
                + half_t_prior(
                    prop_sigma,
                    nu_half_t,
                    taus[j]
                )
                + np.log(prop_sigma)
            )

            if np.log(rng.random()) < (lp_new - lp_old):

                sigmas[j] = prop_sigma

        # =================================================
        # SAVE
        # =================================================

        if it >= burn and ((it - burn) % thin == 0):

            theta_sum += theta
            sigma_sum += sigmas

            keep_count += 1

    # =====================================================
    # POSTERIOR MEAN
    # =====================================================

    theta_mean = (theta_sum / max(1, keep_count)).T

    sigma_mean = sigma_sum / max(1, keep_count)

    loga_mean = theta_mean[:, 0]
    alpha_mean = theta_mean[:, 1]
    logc_mean = theta_mean[:, 2]

    bg_spectra = (
        np.exp(loga_mean)[None, :]
        * (freqs[:, None] ** alpha_mean[None, :])
        + np.exp(logc_mean)[None, :]
    )

    return theta_mean, sigma_mean, bg_spectra

