import numpy as np
from numpy import sqrt
from numpy import exp
from numpy import log


def NAS(delta, epsi, getX):
    t = 0
    mu = 0
    tmp = 1
    while  mu<tmp:
        t = t + 1
        X = getX(1)
        mu = (t-1)/t*mu + X/t
        alpha = sqrt(log(2*t*(t+1)/delta)/(2*t)) # NEWW
        tmp = alpha*(1+1/epsi)
        print(tmp-mu)
    return float(mu), t


def SRA(delta, epsi, getX):
    t = 0
    S = 0
    gamma = 4*(exp(1)-2)*log(2/delta)/epsi**2
    gamma1 = 1+(1+epsi)*gamma
    while  S<=gamma1:
        t = t + 1
        X = getX(1)
        S = S + X
    return gamma1/t, X


def AA(delta, epsi, getX):
    
    gamma = 4*(exp(1)-2)*log(2/delta)/epsi**2
    gamma1 = 2*(1+sqrt(epsi))*(1+2*sqrt(epsi))*(1+log(3/2)/log(2/delta))*gamma
    
    # Step 1
    [mu_prime, X] = SRA(delta/3, min(1/2, sqrt(epsi)), getX)
    
    # Step 2
    N = int(gamma1*epsi/mu_prime) + 1
    X_prime = getX(2*N)
    S = 0
    for i in range(0, N):
        S = S + (X_prime[2*i]-X_prime[2*i+1])**2/2
    rho = max(S/N, epsi*mu_prime)
    t = N

    # Step 3
    N = int(gamma1*rho/mu_prime**2) + 1
    N_remain = N-np.size(X)
    S = sum(X) + sum(getX(N_remain))
    mu = S/N
    t = t+N

    return mu, t




def Update(X, mu, V, t):
    V = (t-1)/(t+1)*V + t/(t+1)**2*(X-mu)**2
    mu = t/(t+1)*mu + X/(t+1)
    return mu, V


def getCt(V, t, d):
    return sqrt(2*V*log(3/d)/t) + 3*log(3/d)/t
    
    
    
    
def EBStopSimple(delta, epsi, getX, p = 1.1):
    t = 2
    X = getX(2)
    mu = np.mean(X)
    V = np.std(X)**2
    tmp = epsi*abs(mu)/(1+epsi)
    c = delta*(p-1)/p
    d = c/t**p
    ct = getCt(V, t, d)
    while  ct>tmp:
        t = t + 1
        X = getX(1)
        d = c/t**p
        [mu, V] = Update(X, mu, V, t)
        tmp = epsi*abs(mu)/(1+epsi)
        ct = getCt(V, t, d)
        print(ct-tmp)
    return float(mu), t






def EBStop(delta, epsi, getX, p = 1.1):
    t = 2
    X = getX(2)
    mu = np.mean(X)
    V = np.std(X)**2
    c = delta*(p-1)/p
    d = c/t**p
    ct = getCt(V, t, d)
    l = abs(mu)-ct
    u = abs(mu)+ct
    while  (1+epsi)*l<(1-epsi)*u:
        t = t + 1
        X = getX(1)
        d = c/t**p
        [mu, V] = Update(X, mu, V, t)
        ct = getCt(V, t, d)
        l = max(l, abs(mu)-ct)
        u = min(u, abs(mu)+ct)
        print((1-epsi)*u - (1+epsi)*l)
    mu_hat = float(np.sign(mu)*((1+epsi)*l+(1-epsi)*u)/2)
    return mu_hat, t





def EBGStop(delta, epsi, getX, beta = 1.1, p = 1.1):
    t = 2
    X = getX(2)
    mu = np.mean(X)
    V = np.std(X)**2
    c = delta*(p-1)/p
    LB = 0
    UB = np.inf
    k = 0
    while  (1+epsi)*LB<(1-epsi)*UB:
        t = t + 1
        X = getX(1)
        [mu, V] = Update(X, mu, V, t)
        if t > np.floor(beta**k):
            k = k + 1
            alpha = np.floor(beta**k)/np.floor(beta**(k-1))
            d = c/k**p
            x = -alpha*log(d/3)
        ct = sqrt(V*2*x/t) + 3*x/t
        LB = max(LB, abs(mu)-ct)
        UB = min(UB, abs(mu)+ct)
        print((1-epsi)*UB - (1+epsi)*LB)
    mu_hat = float(np.sign(mu)*((1+epsi)*LB+(1-epsi)*UB)/2)
    return mu_hat, t


# Betting Algorithms



def getHat(X, sum_X, sum_sq_diff, t):
    sum_X = sum_X + X
    mu_hat = (1/2 + sum_X)/(t+1)
    sum_sq_diff = sum_sq_diff + (X - mu_hat)**2
    sigma_sq_hat = (1/4 + sum_sq_diff)/(t+1)
    return sum_X, mu_hat, sum_sq_diff, sigma_sq_hat



def getKt(Kt, g, X, lambda_tilde, c, ind_LB, ind_UB):
    for i in range(max(ind_LB, 0), ind_UB):
        m = (i+1)/g
        Kt[i, 0] = float(Kt[i, 0]*(1 + min(abs(lambda_tilde), c/m)*(X - m)))
        Kt[i, 1] = float(Kt[i, 1]*(1 - min(abs(lambda_tilde), c/(1-m))*(X - m)))
    return Kt




def Search(Kt_pm, delta, g, ind_LB, ind_UB):
    
    
    for i in range(ind_LB+1, g-1):
        if Kt_pm[i] < 1/delta:
            ind_LB = i-1
            break
    LB = (ind_LB+1)/g
    
    for i in reversed(range(ind_UB)):
        if Kt_pm[i] < 1/delta:
            ind_UB = i+1
            break
    UB = (ind_UB+1)/g
    
    return ind_LB, ind_UB, LB, UB


        


def BettingSimple(delta, epsi, getX, g = 10000):
      
    c = 1/2
    theta = 0.1 
    
    t = 1
    Kt = np.ones((g-1, 2))
    sigma_sq_hat = 1/4 
    lambda_tilde = sqrt((2*log(2/delta))/(sigma_sq_hat*t*log(t+1)))
    
    X = getX(1)
    ind_LB = -1
    ind_UB = g-1
    Kt = getKt(Kt, g, X, lambda_tilde, c, ind_LB, ind_UB)
    Kt_pm = np.maximum(theta*Kt[:, 0], (1-theta)*Kt[:, 1])
    [ind_LB, ind_UB, LB, UB] = Search(Kt_pm, delta, g, ind_LB, ind_UB)
    ct = (UB-LB)/2
    mu_tilde = (UB+LB)/2
    tmp = epsi*abs(mu_tilde)/(1+epsi)
    sum_X, mu_hat, sum_sq_diff, sigma_sq_hat = getHat(X, 0, 0, 1)
    
    
    
    while  ct>tmp:
        t = t+1
        lambda_tilde = sqrt((2*log(2/delta))/(sigma_sq_hat*t*log(t+1)))
        X = getX(1)
        sum_X, mu_hat, sum_sq_diff, sigma_sq_hat = getHat(X, sum_X, sum_sq_diff, t)
        Kt = getKt(Kt, g, X, lambda_tilde, c, ind_LB, ind_UB)
        Kt_pm = np.maximum(theta*Kt[:, 0], (1-theta)*Kt[:, 1])
        [ind_LB, ind_UB, LB, UB] = Search(Kt_pm, delta, g, ind_LB, ind_UB)
        ct = (UB-LB)/2
        mu_tilde = (UB+LB)/2
        tmp = epsi*abs(mu_tilde)/(1+epsi)
        print(ct-tmp)

    return mu_tilde, t




def Betting(delta, epsi, getX, g = 10000):
      
    c = 1/2
    theta = 0.1 
    
    t = 1
    Kt = np.ones((g-1, 2))
    sigma_sq_hat = 1/4 
    lambda_tilde = sqrt((2*log(2/delta))/(sigma_sq_hat*t*log(t+1)))
    
    X = getX(1)
    ind_LB = -1
    ind_UB = g-1
    Kt = getKt(Kt, g, X, lambda_tilde, c, ind_LB, ind_UB)
    Kt_pm = np.maximum(theta*Kt[:, 0], (1-theta)*Kt[:, 1])
    [ind_LB, ind_UB, LB_ct, UB_ct] = Search(Kt_pm, delta, g, ind_LB, ind_UB)
    sum_X, mu_hat, sum_sq_diff, sigma_sq_hat = getHat(X, 0, 0, 1)
    ct = (UB_ct-LB_ct)/2
    mu_tilde = (UB_ct+LB_ct)/2
    LB = float(abs(mu_tilde) - ct)
    UB = float(abs(mu_tilde) + ct)
    mu_tilde = np.sign(sum_X)*((1+epsi)*LB + (1-epsi)*UB)/2
    
    while  (1+epsi)*LB < (1-epsi)*UB:
        t = t+1
        lambda_tilde = sqrt((2*log(2/delta))/(sigma_sq_hat*t*log(t+1)))
        X = getX(1)
        sum_X, mu_hat, sum_sq_diff, sigma_sq_hat = getHat(X, sum_X, sum_sq_diff, t)
        Kt = getKt(Kt, g, X, lambda_tilde, c, ind_LB, ind_UB)
        Kt_pm = np.maximum(theta*Kt[:, 0], (1-theta)*Kt[:, 1])
        [ind_LB, ind_UB, LB_ct, UB_ct] = Search(Kt_pm, delta, g, ind_LB, ind_UB)
        ct = (UB_ct-LB_ct)/2
        mu_tilde = (UB_ct+LB_ct)/2
        LB = max(float(abs(mu_tilde) - ct), LB)
        UB = min(float(abs(mu_tilde) + ct), UB)
        mu_tilde = float(np.sign(mu_tilde)*((1+epsi)*LB + (1-epsi)*UB)/2)
        print((1-epsi)*UB - (1+epsi)*LB)
        
    return mu_tilde, t









