# A Bayesian CWT+MCMC Method for Oscillation Detection in Non-Stationary Time Series

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19446731.svg)](https://doi.org/10.5281/zenodo.19446731)
[![Paper]( https://doi.org/10.3847/1538-4365/ae7345)  
This repository contains the official implementation of the method described in the paper:

> **"A Markov-Chain-Monte-Carlo-based Hybrid Noise Inference for Continuous Wavelet Power Spectra: with Applications to Solar and Stellar Oscillatory Signals"**  
> Feng, S, Li L, and Yuan, D. (2026). *The Astrophysical Journal Supplement Series* (in press).

## Overview

This code provides a hybrid framework that combines the **Continuous Wavelet Transform (CWT)** with **Bayesian MCMC inference** to detect oscillations in solar and stellar time series under non-stationary red noise and evolving background conditions. It overcomes limitations of traditional AR(1)-based wavelet methods and global Fourier approaches.

Key features:
- Time-dependent background spectrum estimation (power-law + white noise)
- Adaptive significance testing without explicit detrending
- Robust detection for Signal-to-Noise Ratio (S/N) ≳ 2
- Works on both synthetic and real GOES soft X-ray flare data


## Requirements

### Python
- numpy, scipy, matplotlib
- PyMC5 (or emcee) for MCMC sampling
- PyWavelets (for CWT)

### IDL
- IDL 8.0+ with built-in wavelet and statistics libraries

- The output significance map indicates where oscillations are detected

If you use this code in your research, please cite:
```
@article{feng2026markov,
  title={A Markov Chain Monte Carlo--based Hybrid Noise Inference for Continuous Wavelet Power Spectra: With Applications to Solar and Stellar Oscillatory Signals},
  author={Feng, Song and Li, Lin and Yuan, Ding},
  journal={The Astrophysical Journal Supplement Series},
  volume={285},
  number={1},
  pages={17},
  year={2026},
  publisher={The American Astronomical Society}
}
```
### Contact

- Song Feng: feng.song@kust.edu.cn
