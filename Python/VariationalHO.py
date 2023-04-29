#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:26:54 2023

@author: johnpaulmbagwu
"""



import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")
import Random as rng


#  define the trial functions
def ftrial0(x,alpha):
    return np.exp(-alpha*x**2)

def ftrial1(x,alpha):
    return x*np.exp(-alpha*x**2)

#  define the local energy functions
def E_local0(x,alpha):
    return alpha+x**2*(1/2-2*alpha**2)

def E_local1(x,alpha):
    return 3*alpha+x**2*(1/2-2*alpha**2)


if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s [-seed 'number'] [-Elevel 'Energy level:' 0,1] [-MCsamples 'Number of MC samples'] [-alpha 'seed for the parameter to estimate'] [-Nalpha 'Number of iterations for minimization'] " % sys.argv[0])
        print
        sys.exit(1)

    # default seed
    seed = 6676
    random = rng.Random(seed)

    # default Energy level 0 
    ftrial = ftrial0
    E_local = E_local0
    Elevel = 0

    #default Number of MC samples
    NMC = 300

    # default seed for alpha
    alpha = 1.5

    # default iterations for alpha
    Nwalk = 40

    # fixed parameters
    Nstep = 300
    gamma = 0.2
    
    # read the user-provided seed from the command line (if there)
    if '-seed' in sys.argv:
        p = sys.argv.index('-seed')
        seed = sys.argv[p+1]
        
    if '-Elevel' in sys.argv:
        p = sys.argv.index('-Elevel')
        Elevel = int(sys.argv[p+1])
        if Elevel == 1:
            ftrial = ftrial1
            E_local = E_local1

    if '-MCsamples' in sys.argv:
        p = sys.argv.index('-MCsamples')
        mc = int(sys.argv[p+1])
        if Ls > 0:
            NMC = mc

    if '-alpha' in sys.argv:
        p = sys.argv.index('-alpha')
        alpha_seed = float(sys.argv[p+1])
        if alpha_seed > 0:
            alpha = alpha_seed

    if '-Nalpha' in sys.argv:
        p = sys.argv.index('-Nalpha')
        iterations = int(sys.argv[p+1])
        if iterations > 0:
            Nwalk = iterations


        
    #define prob distribution
    def PD(x,alpha):
        return ftrial(x,alpha)**2

    #perform the Monte Carlo Integration
    def MonteCarlo(N, alpha):    
        L = 5./np.sqrt(2*alpha) 
        x = random.rand()*2*L-L
        E = 0
        E2 = 0
        Eln = 0
        ln = 0
        for i in range(N):
            xt = random.rand()*2*L-L
            p = PD(xt,alpha)/PD(x,alpha)
            if p >= 1.:
                x = xt
            else:
                if random.rand() < p:
                    x = xt
            E += E_local(x,alpha)
            E2 += E_local(x,alpha)**2
            if Elevel == 0:
                Eln += (E_local(x, alpha)*(-x**2))
                ln += -x**2
            else:
                Eln += (E_local(x, alpha)*(x-x**2))
                ln += x-x**2
        return E/N, E2/N, Eln/N, ln/N

    E_array = np.array([])
    alpha_array = np.array([])
    variance_array = np.array([])

   # Minimization for alpha   
    for i in range(Nwalk):
        E = 0
        E2 = 0
        Eln = 0
        ln = 0
        for j in range(Nstep): 
            E_MC, E2_MC, Eln_MC, ln_MC = MonteCarlo(NMC, alpha)
            E += E_MC/Nstep
            E2 += E2_MC/Nstep
            Eln += Eln_MC/Nstep
            ln += ln_MC/Nstep
        dE_dalpha = 2*(Eln-E*ln)
        print('Alpha: %0.15f' % alpha, '<E>: %0.15f' % E, '\u03C3^2_E: %0.15f' % (E2-E**2))
        alpha = alpha - gamma*dE_dalpha

        E_array = np.append(E_array, E)
        alpha_array = np.append(alpha_array, alpha)
        variance_array = np.append(variance_array, E2-E**2)

        
    # Plots
    plt.grid()
    if Elevel == 1:
        plt.title('HO first excited state energy')
    else:
        plt.title('HO ground state energy')
    plt.plot(E_array, 'b')
    plt.xlabel('Iteration')
    plt.ylabel('<E>')
    plt.show()

    plt.grid()
    if Elevel == 1:
        plt.title('HO first excited state energy')
        plt.ylabel(r'$\beta$')
    else:
        plt.title('HO ground state energy')
        plt.ylabel(r'$\alpha$')
    plt.plot(alpha_array, 'r')
    plt.xlabel('Iteration') 
    plt.show()



    
