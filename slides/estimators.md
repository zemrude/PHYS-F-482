---
marp: true
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



