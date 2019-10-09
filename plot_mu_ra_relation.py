
import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt

plt.style.use(".mplstyle")

ra = np.array([
     11937766,
     16037187,
     21544346,
     28942661,
     38881551,
     52233450,
     70170382,
     94266845,
    126638017,
    170125427,
])

mu = np.array([
     6300,
     6500,
     7900,
     9400,
    10700,
    12700,
    15300,
    19400,
    24900,
    29900,
])

def func(x, a, b, c):
    return a + b*x + c*x**2

a,b,c = opt.curve_fit(func, ra, mu)[0]
x = np.logspace(np.log10(ra.min()), np.log10(ra.max()), 200)
plt.loglog(x, a+b*x+c*x**2)
plt.plot(ra, mu, 'k*')
plt.ylabel(r"$\mu$")
plt.xlabel("Ra")
plt.savefig("mu_ra_relation.pdf", dpi=300)
