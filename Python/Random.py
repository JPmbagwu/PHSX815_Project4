#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:26:54 2023

@author: johnpaulmbagwu
"""

import math
import numpy as np

#################
# Random class
#################
# class that can generate random numbers
class Random:
    """A random number generator class"""

    # initialization method for Random class
    def __init__(self, seed = 5555):
        self.seed = seed
        self.m_v = np.uint64(4101842887655102017)
        self.m_w = np.uint64(1)
        self.m_u = np.uint64(1)
        
        self.m_u = np.uint64(self.seed) ^ self.m_v
        self.int64()
        self.m_v = self.m_u
        self.int64()
        self.m_w = self.m_v
        self.int64()

    # function returns a random 64 bit integer
    def int64(self):
        with np.errstate(over='ignore'):
            self.m_u = np.uint64(self.m_u * np.uint64(2862933555777941757) + np.uint64(7046029254386353087))
        self.m_v ^= self.m_v >> np.uint64(17)
        self.m_v ^= self.m_v << np.uint64(31)
        self.m_v ^= self.m_v >> np.uint64(8)
        self.m_w = np.uint64(np.uint64(4294957665)*(self.m_w & np.uint64(0xffffffff))) + np.uint64((self.m_w >> np.uint64(32)))
        x = np.uint64(self.m_u ^ (self.m_u << np.uint64(21)))
        x ^= x >> np.uint64(35)
        x ^= x << np.uint64(4)
        with np.errstate(over='ignore'):
            return (x + self.m_v)^self.m_w

    # function returns a random floating point number between (0, 1) (uniform)
    def rand(self):
        return 5.42101086242752217E-20 * self.int64()

    # function returns a random integer (0 or 1) according to a Bernoulli distr.
    def Categorical(self, p1=0.25,p2=0.25, p3=0.25, p4=0.25, p5=0.25, p6=0.25):
        if p1 < 0. or p1 > 1.:
            return 1
        if p2 < 0. or p2 > 1.:
            return 2            
        if p3 < 0. or p3 > 1.:
            return 3
        if p4 < 0. or p4 > 1.:
            return 4
        if p5 < 0. or p5 > 1.:
            return 5
        if p6 < 0. or p6 > 1.:
            return 6

                                            
        R = self.rand()

        if R < p1:
         return 1
        if p1< R < p1 + p2:
         return 2
        if p1 + p2  < R < p1+p2+p3:
         return 3
        if p1 + p2 +p3 < R < p1 + p2 + p3 +p4:
         return 4
        if p1 + p2 + p3 + p4  < R < p1 + p2 + p3 + p4 + p5:
         return 5
        if p1 + p2 + p3 + p4  < R < p1 + p2 + p3 + p4 + p5 +p6:
         return 6
        else:
         return 7
     
    # function returns a random double (0 to infty) according to an exponential distribution
    def Exponential(self, beta=1.):
      # make sure beta is consistent with an exponential
        if beta <= 0.:
            beta = 1.

        R = self.rand();

        while R <= 0.:
            R = self.rand()

        X = -math.log(R)/beta

        return X

  
    def Bernoulli(self, p= 0.5):
        if p < 0. or p > 1.:
            return 1
        
        R = self.rand()

        if R < p:
            return 1
        else:

            return 0

    def TruncExp(self, beta, bottom, top):
        a = self.Exponential(beta)
        while (bottom <= a <= top) == False:
            a = self.Exponential(beta)
        return a
