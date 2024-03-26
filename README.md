# Transformers for irregularly sampled time series

Irregular sampling, which is common in astronomy and medicine, rules out most conventional time series methods, as they assume a constant sampling rate. On top of being irregular, the sampling may also be sparse, making learning from these time series even more challenging. This repo contains PyTorch implementations of transformer models that have been proposed to classify irregularly sampled time series. The available architectures are from:

- [Tipirneni and Reddy, "Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series (STraTS)", 2021 ](https://arxiv.org/abs/2107.14293): Only the supervised part, and only the time and value embeddings
- [Astorga et al., "ATAT: Astronomical Transformer for time series And Tabular data", 2023](https://www.researchsquare.com/article/rs-2395110/v1): Only the time series transformer
