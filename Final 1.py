# Plot varying mean and variance

import numpy as np
from numpy import exp
from numpy import sqrt
from numpy import log
from numpy import mean
import scipy.stats as st
import matplotlib.pyplot as plt
import Algorithms as algo

def PlotVarying(n, getX, parameters, mu, R, plot_parameters, delta = 0.1, epsi = 0.1):

    #algorithms = ("Betting", "Betting Simple", "EBGS", "AA", "NAS")
    algorithms = ("Betting", "Betting Simple", "EBGS", "AA")
    t = np.zeros((len(parameters), len(algorithms), n))
    mu_hat = np.zeros((len(parameters), len(algorithms), n))
    test = np.zeros((len(parameters), len(algorithms), n))

    
    for i in range(len(parameters)):
        getX_shrink = lambda k: getX(parameters[i], k)/R[i]
        for k in range(n):
            print([i, n-k])
            [mu_hat[i, 0, k], t[i, 0, k]] = algo.Betting(delta, epsi, getX_shrink)
            [mu_hat[i, 1, k], t[i, 1, k]] = algo.BettingSimple(delta, epsi, getX_shrink)
            [mu_hat[i, 2, k], t[i, 2, k]] = algo.EBGStop(delta, epsi, getX_shrink)
            [mu_hat[i, 3, k], t[i, 3, k]] = algo.AA(delta, epsi, getX_shrink)
            #[mu_hat[i, 4, k], t[i, 4, k]] = algo.NAS(delta, epsi, getX_shrink)
            mu_hat[i, :, k] = mu_hat[i, :, k]*R[i]
            for j in range(len(algorithms)):
                test[i, j, k] = int(epsi*(abs(mu[i]))>abs(mu[i]-mu_hat[i, j, k]))
    
    index = []
    for k in range(n):
        if mean(test[:, :, k]) != 1:
            index.append(k)
    t = np.delete(t, (index), axis = 2)
    t = np.mean(t, axis = 2)
    proba = np.mean(test, axis = 2)
       
    x = np.arange(len(parameters))
    width = 0.15 
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    ax.grid(which = "both", linestyle = ":", axis = "y")
    ax.set_axisbelow(True)
    for i in range(len(algorithms)):
        offset = width * multiplier
        ax.bar(x + offset, t[:, i], width, log = "True", edgecolor = 'black')
        multiplier += 1
    ax.set_ylabel('Average number of samples taken')
    #ax.set_xticks(x + 2*width, plot_parameters)
    ax.set_xticks(x + 1.5*width, plot_parameters)
    #ax.legend(algorithms, loc='upper center', bbox_to_anchor=(0.0000001, -0.15), shadow=False, ncol=3)
    ax.legend(algorithms, loc='upper center', bbox_to_anchor=(0.0000001, -0.15), shadow=False, ncol=2)
    
       
    return proba


def MeanUniform(n):
    
    m = 10
    mu = [0.1, 0.3, 0.5, 0.7, 0.9]
    R = [x + 0.1 for x in mu]
    
    def getX(mu, n):
        a = mu - 0.1
        b = mu + 0.1
        loc = a
        scale = b-a
        X = np.zeros((n, ))
        for i in range(0, n):
            X[i] = mean(st.uniform.rvs(loc = loc, scale = scale, size=(m, )))
        return X
    
    plot_parameters = ['$\mu = 0.1$', '$\mu = 0.3$', '$\mu = 0.5$', '$\mu = 0.7$', '$\mu = 0.9$']
    proba = PlotVarying(n, getX, mu, mu, R, plot_parameters)
    
    return proba

def VarianceUniform(n):
    
    m = [1, 5, 10, 50, 100, 1000]
    a = 0
    b = 1
    mu = (b+a)/2*np.ones((len(m), ))
    R = np.ones((len(m), ))
    
    def getX(m, n):
        loc = a
        scale = b-a
        X = np.zeros((n, ))
        for i in range(0, n):
            X[i] = mean(st.uniform.rvs(loc = loc, scale = scale, size=(m, )))
        return X
    
    plot_parameters = ['$m = 1$', '$m = 5$', '$m = 10$', '$m = 50$', '$m = 100$', '$m = 1000$']
    proba = PlotVarying(n, getX, m, mu, R, plot_parameters)
    
    return proba


def KPut(n):
    
    S0 = 100
    r = 0.01
    sigma = 0.1
    T = 1
    def getX(K, n):
        ST = S0*exp((r-0.5*sigma**2)*T+sigma*sqrt(T)*st.norm.rvs(size=(n,)))
        return exp(-r*T)*(K-ST)*(K>ST)
    
    K = np.array([100, 120, 140, 160])
    R = np.zeros((len(K), ))
    mu = np.zeros((len(K), ))
    for i in range(len(K)):
        d1 = (log(S0/K[i])+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
        d2 = d1-sigma*sqrt(T)
        mu[i] = st.norm.cdf(-d2)*K[i]*exp(-r*T)-st.norm.cdf(-d1)*S0
        R[i] = exp(-r*T)*K[i]
    
    plot_parameters = ['$K = 100$', '$K = 120$', '$K = 140$', '$K = 160$']
    proba = PlotVarying(n, getX, K, mu, R, plot_parameters)
    
    return proba


def KDigitalPut(n):
    
    S0 = 100
    r = 0.01
    sigma = 0.1
    T = 1
    def getX(K, n):
        ST = S0*exp((r-0.5*sigma**2)*T+sigma*sqrt(T)*st.norm.rvs(size=(n,)))
        return exp(-r*T)*(ST<K)
    
    K = np.array([100, 120, 140, 160])
    R = np.zeros((len(K), ))
    mu = np.zeros((len(K), ))
    for i in range(len(K)):
        d1 = (log(S0/K[i])+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
        d2 = d1-sigma*sqrt(T)
        mu[i] = exp(-r*T)*(1 - st.norm.cdf(d2))
        R[i] = exp(-r*T)
    print(mu)
    plot_parameters = ['$K = 100$', '$K = 120$', '$K = 140$', '$K = 160$']
    proba = PlotVarying(n, getX, K, mu, R, plot_parameters)
    
    return proba

n = 30
#proba = MeanUniform(n)
#proba = VarianceUniform(n)
#proba = KPut(n)
proba = KDigitalPut(n)

