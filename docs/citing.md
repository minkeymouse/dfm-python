# Citing

## The software

The citation metadata is in
[CITATION.cff](https://github.com/minkeymouse/dfm-python/blob/main/CITATION.cff);
GitHub's "Cite this repository" button reads it.

## Papers by the author that use dfm-python

Chang, M. and Kim, J.-Y. (2026). Conditionally Identifiable Latent
Representation for Multivariate Time Series with Structural Dynamics.
[arXiv:2603.22886](https://arxiv.org/abs/2603.22886). Introduces iVDFM;
dfm-python is its reference implementation.

Chang, M. (2026). Amortized Neural Filtering for Calibrated Posteriors in
Statistical Arbitrage. 7th ACM International Conference on AI in Finance
(ICAIF '26). [arXiv:2610.25802](https://arxiv.org/abs/2610.25802). Uses AFM
as a baseline; code in [minkeymouse/nfe](https://github.com/minkeymouse/nfe).

```bibtex
@misc{chang2026ivdfm,
  title         = {Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics},
  author        = {Chang, Minkey and Kim, Jae-Young},
  year          = {2026},
  eprint        = {2603.22886},
  archivePrefix = {arXiv}
}

@inproceedings{chang2026amortized,
  title         = {Amortized Neural Filtering for Calibrated Posteriors in Statistical Arbitrage},
  author        = {Chang, Minkey},
  booktitle     = {Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF '26)},
  year          = {2026},
  eprint        = {2610.25802},
  archivePrefix = {arXiv}
}
```

## Method references by model

Please also cite the paper behind the model you use.

| Model | Reference |
|-------|-----------|
| DFM   | Banbura, M. and Modugno, M. (2014), "Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data", *Journal of Applied Econometrics* 29(1). Tent kernel `[1, 2, 3, 2, 1]`: Mariano, R. S. and Murasawa, Y. (2003), "A new coincident index of business cycles based on monthly and quarterly series", *Journal of Applied Econometrics* 18(4). EM acceleration (`em_acceleration="squarem"`): Varadhan, R. and Roland, C. (2008), "Simple and globally convergent methods for accelerating the convergence of any EM algorithm", *Scandinavian Journal of Statistics* 35(2). |
| iVDFM | Chang, M. and Kim, J.-Y. (2026), arXiv:2603.22886 (above). |
| AFM   | Epstein, Wang, Choi and Pelger (2025), "Attention Factors for Statistical Arbitrage", arXiv:2510.11616. LongConv filter: Fu et al. (2023). OU benchmark: Avellaneda and Lee (2010). |
