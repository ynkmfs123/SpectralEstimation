import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pycwt as wavelet
from scipy.stats import chi2
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
from matplotlib.dates import DateFormatter, MinuteLocator

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cwt_mcmc import cwt, mcmc


def load_signal_with_time(data_file):
    time_axis = []
    signal = []

    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            time_str = parts[0]
            value = float(parts[1])

            for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
                try:
                    t = datetime.strptime(time_str, fmt)
                    break
                except ValueError:
                    continue

            time_axis.append(t)
            signal.append(value)

    return np.array(signal), np.array(time_axis), True


def normalize_signal(x):
    return (x - np.mean(x)) / np.std(x)


def pow2_ticks(pmin, pmax):
    emin = int(np.ceil(np.log2(pmin)))
    emax = int(np.floor(np.log2(pmax)))
    return 2 ** np.arange(emin, emax + 1)


def plot_results(time_axis, signal, has_real_time, periods, power, coi, sig_mask):
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    plt.subplots_adjust(wspace=0.3)

    ax1 = axes[0]
    ax1.plot(time_axis, signal, linewidth=2)
    ax1.set_ylabel("Intensity")
    ax1.text(0.02, 0.02, "(a) Signal", transform=ax1.transAxes,
             fontsize=12, ha="left", va="bottom")
    ax1.set_xlim(time_axis[0], time_axis[-1])

    if has_real_time:
        ax1.set_xlabel("Time")
        ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax1.xaxis.set_major_locator(MinuteLocator(byminute=range(0, 60, 5)))
    else:
        ax1.set_xlabel("Time (s)")

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax1.yaxis.set_major_formatter(formatter)
    ax1.yaxis.offsetText.set_fontsize(10)

    ax2 = axes[1]
    log_power = np.log2(power)
    vmin, vmax = np.percentile(log_power, [5, 95])
    ticks_2pow = pow2_ticks(periods.min(), periods.max())
    coi_period = np.clip(coi, periods.min(), periods.max())
    levels = np.linspace(vmin, vmax, 25)

    ax2.contourf(
        time_axis,
        periods,
        log_power,
        levels=levels,
        cmap=plt.cm.coolwarm,
        vmin=vmin,
        vmax=vmax,
        extend="both",
    )
    ax2.contour(
        time_axis,
        periods,
        sig_mask.astype(float),
        levels=[0.5],
        colors="k",
        linewidths=1.3,
    )
    ax2.plot(time_axis, coi_period, "k--", lw=1.2)
    ax2.fill_between(
        time_axis,
        coi_period,
        periods.max(),
        where=coi_period < periods.max(),
        facecolor="none",
        edgecolor="gray",
        hatch="x",
        alpha=0.7,
        zorder=3,
    )
    ax2.set_yscale("log", base=2)
    ax2.set_ylim(periods.min(), periods.max())
    ax2.set_yticks(ticks_2pow)
    ax2.set_yticklabels([f"{int(v)}" for v in ticks_2pow])
    ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.tick_params(axis="y", which="minor", length=0)
    ax2.invert_yaxis()
    ax2.set_ylabel("Period (s)")
    ax2.text(0.02, 0.02, "(b) CWT - MCMC", transform=ax2.transAxes,
             fontsize=12, ha="left", va="bottom")
    ax2.set_xlim(time_axis[0], time_axis[-1])

    if has_real_time:
        ax2.set_xlabel("Time")
        ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(MinuteLocator(byminute=range(0, 60, 5)))
    else:
        ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def main():
    data_file = "../synthetic_signal.txt"
    dt = 2.0

    signal, time_axis, has_real_time = load_signal_with_time(data_file)
    x_original = normalize_signal(signal)

    mother = wavelet.Morlet(6)
    dj = 1 / 24
    s0 = 2 * dt
    j = -1

    power, freqs, coi, periods = cwt(
        x=x_original,
        dt=dt,
        mother=mother,
        dj=dj,
        s0=s0,
        j=j,
    )

    seed = 1234
    n_iter = 20000
    burn_frac = 0.25
    thin = 2
    signif_level = 0.95

    mu_anchor = np.array([-8, -2, -4], dtype=float)
    sd_anchor = np.array([2, 1, 2], dtype=float)

    nu_half_t = 3.0
    tau_loga = 0.3
    tau_alpha = 0.1
    tau_logc = 0.4

    prop_sig_loga = 0.05
    prop_sig_alpha = 0.02
    prop_sig_logc = 0.05
    prop_sig_logsig = 0.1

    theta_means, sigmas_mean, bg_spectra = mcmc(
        y=power,
        freqs=freqs,
        n_iter=n_iter,
        burn_frac=burn_frac,
        thin=thin,
        seed=seed,
        mu_anchor=mu_anchor,
        sd_anchor=sd_anchor,
        nu_half_t=nu_half_t,
        tau_loga=tau_loga,
        tau_alpha=tau_alpha,
        tau_logc=tau_logc,
        prop_sig_loga=prop_sig_loga,
        prop_sig_alpha=prop_sig_alpha,
        prop_sig_logc=prop_sig_logc,
        prop_sig_logsig=prop_sig_logsig,
    )

    factor = chi2.ppf(signif_level, 2) / 2
    sig_mask = power > (bg_spectra * factor)

    plot_results(
        time_axis=time_axis,
        signal=signal,
        has_real_time=has_real_time,
        periods=periods,
        power=power,
        coi=coi,
        sig_mask=sig_mask,
    )


if __name__ == "__main__":
    main()
