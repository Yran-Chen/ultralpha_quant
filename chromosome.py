# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 18:24:33 2019

@author: ultralpha
"""

import numpy as np
import pandas as pd

class Chromosome:
    def __init__(self, no_of_assets):
        self.no_of_assets = no_of_assets
#        self.assets = 0
        self.weights = 0
        self.chromosome = np.random.rand(no_of_assets)
        self.to_replace = False
        self.fitness = -np.inf

#        self.portfolio_prices = 0
#        self.validation_portfolio_prices = 0
#        self.test_portfolio_prices = 0
    
    def mutate(self, mutrate):
#       mutrate is the possibility of mutrate
        for i in range(self.no_of_assets):
            if np.random.random() < mutrate:
                self.chromosome[i] = np.random.random()     
                
                
    def clone(self):
        cln = Chromosome(self.no_of_assets)

        for i in range(self.no_of_assets):
            cln.chromosome[i] = self.chromosome[i]
            
        if len(self.weights) > 0:
            cln.weights = np.zeros(len(self.weights))
#            cln.assets = np.zeros(len(self.weights))

        for i in range(len(self.weights)):
            cln.weights[i] = self.weights[i]
#            cln.assets[i] = self.assets[i]
        cln.to_replace = self.to_replace
        return cln