# Plot varying delta and epsilon

import numpy as np
from numpy import log
from numpy import mean
import scipy.stats as st
import matplotlib.pyplot as plt
import Algorithms as algo

a = 0
b = 1
mu = (b+a)/2 
R = b
# m = 2
m = 4
var = (b-a)**2/(12*m)

def getX(n):
    X = np.zeros((n, ))
    for i in range(0, n):
        X[i] = mean(st.uniform.rvs(loc = a, scale = b-a, size=(m, )))
    return X



def PlotDelta(n):

    epsi = 0.1
    delta = np.array([10**(-(4*i + 1)) for i in range(4)])
    algorithms = ("Betting", "Betting Simple", "EBGS", "AA", "NAS")
    t = np.zeros((len(delta), len(algorithms), n))
    mu_hat = np.zeros((len(delta), len(algorithms), n))
    test = np.zeros((len(delta), len(algorithms), n))
    
    for i in range(len(delta)):
        for k in range(n):
            print([i, n-k])
            [mu_hat[i, 0, k], t[i, 0, k]] = algo.Betting(delta[i], epsi, getX)
            [mu_hat[i, 1, k], t[i, 1, k]] = algo.BettingSimple(delta[i], epsi, getX)
            [mu_hat[i, 2, k], t[i, 2, k]] = algo.EBGStop(delta[i], epsi, getX)
            [mu_hat[i, 3, k], t[i, 3, k]] = algo.AA(delta[i], epsi, getX)
            [mu_hat[i, 4, k], t[i, 4, k]] = algo.NAS(delta[i], epsi, getX)
            for j in range(len(algorithms)):
                test[i, j, k] = int(epsi*(abs(mu))>abs(mu-mu_hat[i, j, k]))
                
    index = []
    for k in range(n):
        if mean(test[:, :, k]) != 1:
            index.append(k)
    t = np.delete(t, (index), axis = 2)
    t = np.mean(t, axis = 2)
    proba = np.mean(test, axis = 2)
     
    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(algorithms)):
        ax.loglog(delta, t[:, i])
    ax.loglog(delta, -100*log(delta))
    ax.legend(("Betting", "Betting Simple", "EBGS", "AA", "NAS","C log"r'$(\delta)$') , fontsize = 11,loc='upper center', bbox_to_anchor=(0.0000001, -0.15), shadow=False, ncol=3)
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel('Average number of samples taken')
    
       
    return proba







def PlotEpsilon(n):
    

    delta = 0.1
    epsi_lim = var/(mu*R)
    epsi = np.sort(np.array([epsi_lim, 0.1, 0.01]))
    algorithms = ("Betting", "Betting Simple", "EBGS", "AA", "NAS")
    t = np.zeros((len(epsi), len(algorithms), n))
    mu_hat = np.zeros((len(epsi), len(algorithms), n))
    test = np.zeros((len(epsi), len(algorithms), n))
    
    for i in range(len(epsi)):
        for k in range(n):
            print([i, n-k])
            [mu_hat[i, 0, k], t[i, 0, k]] = algo.Betting(delta, epsi[i], getX)
            [mu_hat[i, 1, k], t[i, 1, k]] = algo.BettingSimple(delta, epsi[i], getX)
            [mu_hat[i, 2, k], t[i, 2, k]] = algo.EBGStop(delta, epsi[i], getX)
            [mu_hat[i, 3, k], t[i, 3, k]] = algo.AA(delta, epsi[i], getX)
            [mu_hat[i, 4, k], t[i, 4, k]] = algo.NAS(delta, epsi[i], getX)
            for j in range(len(algorithms)):
                test[i, j, k] = int(epsi[i]*(abs(mu))>abs(mu-mu_hat[i, j, k]))
    t = np.mean(t, axis = 2)
    proba = np.mean(test, axis = 2)
     
    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(algorithms)):
        ax.loglog(epsi, t[:, i])
    ax.axvline(x = epsi_lim, linestyle = "dashed")
    plt.loglog(epsi, 600*epsi**(-1.0))
    plt.loglog(epsi, 2*epsi**(-2.0))
    ax.legend(("Betting", "Betting Simple", "EBGS", "AA", "NAS", "Threshold", r'$\epsilon^{-1}$', r'$\epsilon^{-2}$') , fontsize = 11,loc='upper center', bbox_to_anchor=(0.0000001, -0.15), shadow=False, ncol=4)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel('Average number of samples taken')
    
    return proba



n = 30
#proba = PlotDelta(n)
proba = PlotEpsilon(n)







