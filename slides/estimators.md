---
marp: true
math: mathjax
---
# PHYS-F-482: Advance techniques in experimental physics

## Theory of Estimators 
Juan A. Aguilar 

---

# Table of Content
* Introduction of estimator
* Propierties
* Examples of estimators
* Likelihood method
* ...
  
---

# Parameter estimation
The parameters of a distribution are constants that characterize their shape:

$$ f(x; \theta) = \frac{1}{\theta}\exp^{-\frac{x}{\theta}}$$
where $x$ is a r.v. and $\theta$ is a *parameter*.

Suppose we have a sample of observed values $\vec{x} = (x_1,x_2, ...)$ following the $f(x;\theta)$ distribution. We want to some function of the data $\vec{x}$ that **estimates** the value of the parameter $\theta$. 

  $$\hat{\theta}(\vec{x})$$
 
We put a *hat* to say that this is an estimator. We sometimes called *estimator* to the function, and *estimate* to the value that comes out with particular data set.

---

# Properties of Estimators

Estimates of $\hat{\theta}(\vec{x})$ depend on the r.v. $\vec{x}$, therefore, estiamtes are also r.v. and they follow an specific *pdf* $g(\hat{\theta}; \theta)$ that generally depends on the true value $\theta$:

![estimators](figs/estimators.png)

* The **bias** of an estimator is defined as: $b = E[\hat{\theta}] - \theta$
  * Average repeated measurements should tend to 0 bias
* The **variance** is $V[\hat{\theta}] = E[\hat{\theta}^2]- (E[\hat{\theta}])^2$
  * Small variance 

---
# Properties of Estimators
Another measure of the quality of an estimator is called the *MSE* (Mean Squared error):
$$MSE = E[(\hat{\theta}- \theta)^2] = E[(\hat{\theta} - E[\hat{\theta}])^2]+(E[\hat{\theta}-\theta])^2 = V[\hat{\theta}]-b^2$$

:point_right: Small bias and variance are in general conflicting criteria. An estimator is called optimal if its bias is 0 and the variance minimal. 

---
# Example of estimator I

* Parameter: $\mu = E[x]$

* Estimator: $\hat{\mu} = \frac{1}{n}\sum_{i = 1}^n x_i \equiv \bar{x}$ 

> **Bias**: $b = E[\hat{\mu}] - \mu = \frac{1}{n}E[x_1 + x_2 + ... + x_n] - \mu = \frac{1}{n}E[n\mu] -\mu = 0$ 
> 
> **Variance**: $V[\hat{\mu}] = V[\frac{1}{n}x_1 +  ... + \frac{1}{n}x_n] = \frac{1}{n^2}\left(V[x_1] +  ... + V[x_n]\right) = \frac{1}{n^2}(n\sigma^2) = \frac{\sigma^2}{n}$

---
# Example of estimator II

* Parameter: $\sigma^2 = V[x]$
* Estimator: $\hat{\sigma}^2 = \frac{1}{1 - n^2}\sum_{i = 1}^{n}(x_i - \bar{x})^2 \equiv s^2$

> **Bias**: $E[\hat{\sigma}^2] - \sigma^2 = 0$ (the factor $n-1$ makes it possible)
>
> **Variance**: $V[\hat{\sigma}^2] = \frac{1}{n}(\mu_4 - \frac{n-3}{n-1}\mu_2),$ where $\mu_k = \int (x-\mu)^k f(x) {\rm d}x$

--- 

# The Likelihood Function
Suppose a set of measurements $x_i$ each independent and identically distributed (i.i.d) ie, each follows a probability distribution $f(x;\vec{\theta})$ that depends on a set parameter $\vec{\theta}$. 

If we evaluate the function with the data obtained and regard it as a function of the parameters $\vec{\theta}$ this is called the **likelihood funcion**:

$$\mathcal{L}(\vec{\theta}) = f(\vec{x};\vec{\theta}) = \prod_{i=1}^n f(x_i;\vec{\theta})$$
where $x_i$ are constants

---

# The Maximum Likelihood Method

The likelihood function is a function of $\vec{\theta}$ if we choose a $\theta$ close to the true value, it is expected that the probabilities are high.  So we define the maximum likelihood (ML) estimators to be the parameters that maximize the likelihood. If the likelihood function is differentiable the estiamtors are given by:

$$ \frac{\partial \mathcal{L}}{\partial \theta_i} = 0$$
---
# ML method: Example I

Consider the decay time of a particle, which is given by the exponential *pdf* $f(t; \tau) = \frac{1}{\tau}e^{-{t/\tau}}$ where $\tau$ is the lifetime of the particle. 
Imagine we have a set of measurements for different decays $t_1, ..., t_n$, the likelihood would be:
$$\mathcal{L}(\tau) = \prod_{i =1 }^n \frac{1}{\tau}e^{-\frac{t_i}{\tau}}$$

The value of $\tau$ for which $\mathcal{L}(\tau)$ is maximum also gives the maximum value of its logarithm (the log-likelihood function):

$$\log \mathcal{L}(\tau)  = \sum_{i =1 }^n \log \frac{1}{\tau}e^{-\frac{t_i}{\tau}}  = \sum_{i =1 }^n \left(\log{\frac{1}{n}} - \frac{t_i}{\tau}\right)$$


---

# ML method: Example I

Finding the maximum $\frac{\partial \log \mathcal{L}(\tau)}{\partial\tau} = 0$ gives:
$$ \hat{\tau} = \frac{1}{n}\sum_{i=1}^n t_i$$


```python
%matplotlib inline
import numpy as np
import scipy as sp
from scipy import special
from scipy.stats import expon

import matplotlib.pylab as plt

from IPython.display import Markdown
```
---
# ML method: Example I
```python
def exp(x, tau):
    return 1/tau * np.exp(x/tau)
tau  = 1.

x = np.linspace(0, 10, 1000)
y = exp(x, tau)

nevents = 50
data = np.random.exponential(tau, nevents)

fig, ax = plt.subplots(figsize=(8,6)) 
ax.scatter(data, np.zeros(nevents), s = 500, marker = '|') 
ax.plot(x, expon(0,tau).pdf(x), lw=2, color="red")
ax.set_xlim(0,5)
ax.set_ylim(0,1)
```
---
# ML method: Example I
![tau_distribution](./figs/tau_distribution.png)
```python
tau_estimate =  1./nevents * np.sum(data)
```
Best estimate is $\hat{\tau} = 1.01$

---
# ML method: Example II
Let's consider the case of a gaussian distribution:
$$f(x;\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x-\mu)^2/2\sigma^2}$$
If we have $x_1,...,x_n$ independent randon variables that follow the gaussian distribution,the log-likelihood is given by:
$$\log \mathcal{L}(\mu, \sigma^2) = \sum_{i=1}^n \log f(x_i;\mu,\sigma^2) = \sum_{i=1}\left(\log\frac{1}{\sqrt{2\pi}}+ \frac{1}{2}\log\frac{1}{\sigma^2}- \frac{(x_i - \mu)^2}{2\sigma^2}\right)$$
---
# ML method: Example II
Calculating the derivatives for $\mu$ and $\sigma^2$ we have:
$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i$$
and
$$\hat{\sigma^2}= \frac{1}{n}\sum_{i=1}^n(x_i- \mu)^2$$
---
# ML method: Example II
* The estimator for $\mu$ is unbiased, but we find that $E[\hat{\sigma^2}] = \frac{n - 1}{n}\sigma^2$, so the ML estiamtor for $\sigma^2$ has a bias, that disappears when $n \rightarrow \infty$.
* To have an unbiased estimator we can use:
  $$ s^2 = \frac{1}{n - 1}\sum_{i=1}^n(x_i- \mu)^2$$
---
# Variance of Estimators
Once we have defined the estimators, we want to report its _statistical error_. I.e., how widely the estimate will distribute if we repeat the measurement many times. We are going to see 4 methods:
1. Analytical (when possible)
2. Monte Carlo method
3. Using the information inequality
4. Graphical Method

---

# Variance of Estimators: Analytical 
In some cases we can calculate the variance analytically. For example for the exponential distribution.
* We found the estimator as:  $\hat{\tau} = \frac{1}{n}\sum_{i=1}^n t_i$
* Variance is defined as: $V[\hat{\tau}] = E[\hat{\tau^2}] - (E[\hat{\tau}])^2$

$$=\int...\int\left(\frac{1}{n}\sum_{i=1}^n t_i\right)^2\frac{1}{\tau}e^{-t_1/\tau}...\frac{1}{\tau}e^{-t_n/\tau} {\rm d}t_1...{\rm d}t_n - \left(\int...\int\left(\frac{1}{n}\sum_{i=1}^n t_i\right)\frac{1}{\tau}e^{-t_1/\tau}...\frac{1}{\tau}e^{-t_n/\tau} {\rm d}t_1...{\rm d}t_n\right)^2$$
$$= \frac{\tau^2}{n}$$
* Note that the variance depends on $\tau$ the __true value__. In practice we take $\hat{\tau}$ as the value for the variance.
---
# Variance of Estimators: Monte Carlo
In several cases, we cannot calcualte the variance analytically. In those cases we can use a Monte Carlo method:

```python
nexperiments = 100
tau_estimates = []
for i in range(0, nexperiments):
   data = np.random.exponential(tau, nevents)
   tau_estimates.append(1./nevents * np.sum(data))
```
---
# Variance of Estimators: Monte Carlo
![](./figs/tau_estimate_distribution.png)



