# JAWS-X

This repository contains code for both a 2023 ICML paper a 2023 COPA extended abstract:

1. Drew Prinster, Suchi Saria, and Anqi Liu. JAWS-X: Addressing efficiency bottlenecks of conformal prediction under standard and feedback covariate shift. *International Conference on Machine Learning (ICML)*, 2023.

2. Drew Prinster, Suchi Saria, and Anqi Liu. Efficient Approximate Predictive Inference Under Feedback Covariate Shift with Influence Functions. *Conformal and Probabilistic Prediction with Applications (COPA)*, 2023.*

We also build heavily on code from the following paper: Clara Fannjiang, Stephen Bates, Anastasios N Angelopoulos, Jennifer Listgarten, and Michael I Jordan. Conformal prediction under feedback covariate shift for biomolecular design. *Proceedings of the National Academy of Sciences*, 119(43):e2204569119, 2022.

This repository was last updated and cleaned up on August 11th, 2023! The code should now be ready to clone and play with, though I will continue to work improving the repo's useability and clarity throughout September 2023. Please don't hestitate to reach out at **drew@cs.jhu.edu** with questions!

---

*Code and experimental details for "Efficient Predictive Interval Approximation Under Feedback Covariate Shift with Higher-Order Influence Functions" (COPA 2023):

For the main experiment presented in Figure 1 of the extended abstract, a neural network predictor $\widehat{\mu}$ with one hidden layer containing 25 neurons, tanh activation function, L2 regularization strength value 0.5, and trained for 2000 epochs. Designed protein sequences were sampled in proportion to $\exp(\lambda\cdot \widehat{\mu}(X_{\text{test}}))$, where $\lambda$ is a tuning parameter that increases with shift magnitude. Figure 1 reports mean coverage, mean predicted fitness, and median interval width across 20 random seeds and random training sets. Error bars are standard error for mean coverage and mean predicted fitness, and are upper and lower quartiles for median interval width.
