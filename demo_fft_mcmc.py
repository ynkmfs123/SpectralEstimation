import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
from matplotlib.dates import DateFormatter, MinuteLocator

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fft_mcmc import fft, mcmc

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


def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)


def plot_results(time_axis, signal, has_real_time, period, pxx, log_s_fit):
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
    log_period = np.log(period)
    ax2.plot(log_period, np.log(pxx), lw=1.2, label="PSD")
    ax2.plot(log_period, log_s_fit, "r", lw=2, label="MCMC fit")
    ax2.plot(log_period, log_s_fit + 1.09, "r--", lw=1, label="95%")

    tick_vals = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    ax2.set_xticks(np.log(tick_vals))
    ax2.set_xticklabels([str(v) for v in tick_vals])
    ax2.set_xlim(log_period.max(), log_period.min())
    ax2.set_xlabel("Period (s)")
    ax2.set_ylabel("Fourier power")
    ax2.text(0.02, 0.02, "(b) PSD(log)", transform=ax2.transAxes,
             fontsize=12, ha="left", va="bottom")
    ax2.legend(loc="upper right", fontsize=10, frameon=True)

    plt.tight_layout()
    plt.show()


def main():
    data_file = "synthetic_signal.txt"
    dt = 2.0

    signal, time_axis, has_real_time = load_signal_with_time(data_file)
    signal_norm = normalize_signal(signal)

    n = len(signal_norm)
    window = np.hanning(n)
    f, pxx = fft(signal_norm, dt, window)

    log_s_fit, theta, post, chain = mcmc(
        f=f,
        psd=pxx,
        n_iter=4000,
        mu=np.array([-10.0, -2.0, -5.0]),
        prior_sig=np.array([2.0, 1.0, 2.0]),
        prop_sig=np.array([2.0, 1.0, 2.0]),
        burn=1000,
        thin=2,
    )

    period = 1 / f



    plot_results(
        time_axis=time_axis,
        signal=signal,
        has_real_time=has_real_time,
        period=period,
        pxx=pxx,
        log_s_fit=log_s_fit,
    )


if __name__ == "__main__":
    main()