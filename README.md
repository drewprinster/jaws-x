# jaws-x

This repository contains code for two papers:
(1) JAWS-X: Addressing Efficiency Bottlenecks of Conformal Prediction Under Standard and Feedback Covariate Shift, ICML, 2023.
(2) Efficient Approximate Predictive Inference Under Feedback Covariate Shift with Influence Functions, COPA, 2023.

Cleaned-up code will be updated to the repository by **August 11th, 2023**. Thank you for your patience, as I've been traveling and moving to a new home!

---

Code and experimental details for "Efficient Predictive Interval Approximation Under Feedback Covariate Shift with Higher-Order Influence Functions":

For the main experiment presented in Figure 1 of the extended abstract, a neural network predictor $\widehat{\mu}$ with one hidden layer containing 25 neurons, tanh activation function, L2 regularization strength value 0.5, and trained for 2000 epochs. Designed protein sequences were sampled in proportion to $\exp(\lambda\cdot \widehat{\mu}(X_{\text{test}}))$, where $\lambda$ is a tuning parameter that increases with shift magnitude. Figure 1 reports mean coverage, mean predicted fitness, and median interval width across 20 random seeds and random training sets. Error bars are standard error for mean coverage and mean predicted fitness, and are upper and lower quartiles for median interval width.
