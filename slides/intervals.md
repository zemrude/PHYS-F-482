---
marp: true
math: mathjax
paginate: true

---

# PHYS-F-482: Advance techniques in experimental physics

## Confidence Intervals 

Juan A. Aguilar, Pascal Vanlaer

(Based on Cowen Lectures)

---

# Interval Estimation

* In addition of giving a *point estimate* of parameter we should report an **interval** reflecting the statistical uncertainty. 
* Usually the **variance of the estimator** is used (as $\hat{\theta} \pm \hat{\sigma}_\theta$).
* BUT sometimes this is not adequate (for example wehn the result is close to physical boundary):
  ![](/figs/estimate_distribution_negvalues.png)

---

# Frequentist confidence interval

* Let's consider $\hat{\theta}$ and estimator of the parameter $\theta$, $\hat{\theta}_{obs}$ an estimate.
* We know the estiamtes are distributed as $g(\hat{\theta}, \theta)$. In principle we need to know this distribution for every possible value of $\theta$
* We define an upper and lower limit tail probabilities, e.g.  $\alpha = 0.05$, $\beta = 0.05$. Then we need to find the functions $u_\alpha(\theta)$, $u_\beta(\theta)$ such as:
  
  $$ \alpha = P(\hat{\theta})