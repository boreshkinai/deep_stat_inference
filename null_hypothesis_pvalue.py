import numpy as np
import numpy.random as nprnd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

N = 100
M = 20000
PERCENTILE_RANGE = np.arange(0, 101)

mu = 1.5
sigma = 3.0

X = nprnd.normal(0, sigma, (N, M))
Y = nprnd.normal(mu, sigma, (N, M))

Z1 = nprnd.normal(0, sigma, (N, M))
Z2 = nprnd.normal(0, sigma, (N, M))

# H0 model parameter estimation under H0:
muH0 = np.mean(np.concatenate([Z1, Z2], axis=0), axis=0)
sigmaH0 = np.sqrt(np.var(np.concatenate([Z1, Z2], axis=0), axis=0))
# H1 model parameter estimation under H0:
muZ1H1 = np.mean(Z1, axis=0)
muZ2H1 = np.mean(Z2, axis=0)
sigmaZ1H1 = np.sqrt(np.var(Z1, axis=0))
sigmaZ2H1 = np.sqrt(np.var(Z2, axis=0))

TZ = np.sum(stats.norm.logpdf(Z1, loc=muZ1H1, scale=sigmaZ1H1) + stats.norm.logpdf(Z2, loc=muZ2H1, scale=sigmaZ2H1) -
            stats.norm.logpdf(Z1, loc=muH0, scale=sigmaH0) - stats.norm.logpdf(Z2, loc=muH0, scale=sigmaH0), axis=0)
percentileH0 = np.percentile(TZ, PERCENTILE_RANGE)
ecdfH0 = interp1d(percentileH0, PERCENTILE_RANGE, fill_value=(0.0, 100.0), bounds_error=False)
p1 = np.sum(ecdfH0(TZ) <= 5) / M

# H0 model parameter estimation under H0:
muXYH0 = np.mean(np.concatenate([X, Y], axis=0), axis=0)
sigmaXYH0 = np.sqrt(np.var(np.concatenate([X, Y], axis=0), axis=0))
# H1 model parameter estimation under H0:
muXH1 = np.mean(X, axis=0)
muYH1 = np.mean(Y, axis=0)
sigmaXH1 = np.sqrt(np.var(X, axis=0))
sigmaYH1 = np.sqrt(np.var(Y, axis=0))

TXY = np.sum(stats.norm.logpdf(X, loc=muXH1, scale=sigmaXH1) + stats.norm.logpdf(Y, loc=muYH1, scale=sigmaYH1) -
            stats.norm.logpdf(X, loc=muXYH0, scale=sigmaXYH0) - stats.norm.logpdf(Y, loc=muXYH0, scale=sigmaXYH0), axis=0)
pValues = 100 - ecdfH0(TXY)
pXY = np.sum(pValues <= 5) / M

plt.plot(TZ)
plt.plot(TXY)
plt.show()

plt.hist(pValues, 100)
plt.show()
