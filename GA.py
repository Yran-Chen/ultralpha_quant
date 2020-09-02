# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:50:59 2019

@author: ultralpha
"""

import numpy as np
import pandas as pd
import time
from chromosome import Chromosome
from performace_evaluator import Performace_evaluator

class GeneticAlgorithm:
    prob_crossover, replacement_rate, prob_mutation, tournament_size = None, None, None, None

    @classmethod
    def update_params(cls, prob_crossover, replacement_rate, prob_mutation, tournament_size):
        cls.prob_crossover = prob_crossover
        cls.replacement_rate = replacement_rate
        cls.prob_mutation = prob_mutation
        cls.tournament_size = tournament_size

    def __init__(self, strategy_ret, ga_cycles = 100, population_size = 100, genetic_params=[0.5, 0.3, 0.1, 100], evaluate_params=[]):
        '''
        Args:
           genetic_params (List[int])
                A list of parameters that will be used to construct GA 

           evaluate_params (List[int])
                List of parameters to evaluate the performance of the weights
        '''
        prob_crossover, replacement_rate, prob_mutation, tournament_size = genetic_params
        self.update_params(prob_crossover, replacement_rate, prob_mutation, tournament_size)

        self.strategy_ret = strategy_ret
        self.strategy_pool = strategy_ret.columns
        self.ga_cycles = ga_cycles

        
#       evaluate function has not been defined
#        self.evaluator = Performace_evaluator()
        self.Genetics = Genetics(population_size, len(self.strategy_pool), self.__class__)

    def run(self,target = "Mean_variance"):

        for i in range(self.ga_cycles):
            if i > 0:
                self.Genetics.get_population()
            fit_box = []
            for i in range(len(self.Genetics.genes)):
                #                
                #
                #最后这个输入要换成  1.策略收益的dataframe  2.该表现型预设置的权重
                #
                #
                x = Performace_evaluator(self.Genetics.genes[i].chromosome, self.strategy_ret)
                scores = x.run(target)
                # fit_box[i] = scores[0]
                self.Genetics.genes[i].fitness = scores[0]
#                self.Genetics.genes[i].pfreturns = scores[1]
                self.Genetics.genes[i].weights = scores[1]
#                self.Genetics.genes[i].assets = scores[2]


        for i in range(len(self.Genetics.genes)):
            x = Performace_evaluator(self.Genetics.genes[i].chromosome, self.strategy_ret)
            scores = x.run(target)
#            scores = self.evaluator.evaluate(self.Genetics.genes[i].chromosome, validate_interval)
            self.Genetics.genes[i].validfitness = scores[0]
#            self.Genetics.genes[i].validpfreturns = scores[1]

    def run_softmax(self,loss_weight, target="test_"):
            start = time.clock()
            counter = 0
            total_len = self.ga_cycles
            for i in range(self.ga_cycles):
                if i > 0:
                    self.Genetics.get_population()
                fit_box = []
                for j in range(len(self.Genetics.genes)):

                    x = Performace_evaluator(self.Genetics.genes[j].chromosome, self.strategy_ret,loss_weight=loss_weight)
                    scores = x.run(target)
                    fit_box.append(scores[0])
                    self.Genetics.genes[j].weights = scores[1]

                fit_box_np = np.array(fit_box).reshape(len(self.Genetics.genes), -1)
                pt_box = np.array([self.softmax(fit_box_np[:,guideline]) for guideline in range(fit_box_np.shape[1])]).sum(axis=0)

                for i_ in range(len(self.Genetics.genes)):
                    self.Genetics.genes[i_].fitness = pt_box[i_]

                counter += 1
                print('{}/{}'.format(counter,self.ga_cycles))
                elapsed = (time.clock() - start)
                total_time = (elapsed / (counter) * (total_len))
                print('Time processed remained : {:.2f}/{:.2f}'.format(elapsed, total_time))

            fit_box = []
            for i in range(len(self.Genetics.genes)):
                x = Performace_evaluator(self.Genetics.genes[i].chromosome, self.strategy_ret,loss_weight=loss_weight)
                scores = x.run(target)
                fit_box.append(scores[0])

            fit_box_np = np.array(fit_box).reshape(len(self.Genetics.genes), -1)
            pt_box = np.array([self.softmax(fit_box_np[:,i]) for i in range(fit_box_np.shape[1])]).sum(axis=0)

            for i in range(len(self.Genetics.genes)):
                    self.Genetics.genes[i].validfitness = pt_box[i]
                #            scores = self.evaluator.evaluate(self.Genetics.genes[i].chromosome, validate_interval)
                # self.Genetics.genes[i].validfitness = scores[0]

    #            self.Genetics.genes[i].validpfreturns = scores[1]

    def softmax(self,df):
        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)
        return softmax(df)

    def best_portfolio(self, portfoliotype):
        '''
        Args:
            portfoliotype (str)
                The sample where to take the returns from: ['train','valid','test']
        '''
        if portfoliotype is None:
            portfoliotype = 'test'
        
        #best is a chromosome class
        best = self.Genetics.best_gene()
        return  best.chromosome ,(best.chromosome / best.chromosome.sum())
        # return  best.chromosome / best.chromosome.sum()

    def solution(self):
        return self.Genetics.best_gene()

class Genetics:
    def __init__(self, population_size, no_of_assets, ga_type):
        '''
        Args:
            population_size (int)
                Number of genes
            no_of_assets (int)
                Number of available stocks
            ga_type (GeneticAlgorithm)
                using this to access the global variables of the class GeneticAlgorithm
        '''
        self.population_size = population_size
        self.no_of_assets = no_of_assets
        self.genes = [Chromosome(no_of_assets) for i in range(population_size)]

        self.fittest_genes = []
        self.unfittest_genes = []
        self.fittest_index = 0 # index of fittest chromosome
        self.ga_type = ga_type

    def best_gene(self):
        validation_scores = list(map(lambda i: i.validfitness, self.genes))
        self.fittest_gene = np.argmin(validation_scores)
        return self.genes[self.fittest_gene]

    def get_population(self):
        self.tournament()
        self.crossover()
        self.mutate()

    def tournament(self):
        self.fittest_genes = []
        self.unfittest_genes = []
# if replace is true, it will be appended into the unfittest group
# firstly, all of the gene will be appended into the unfittest group
        for i in range(len(self.genes)):
            self.genes[i].replace=True
            self.unfittest_genes.append(self.genes[i])

# find the minimun of fitness its self.genes, in this program, 
# the smaller the fitness, the better the gene is.            
        ftns_thres = np.inf
        for i in range(len(self.genes)):
            if self.genes[i].fitness < ftns_thres:
                ftns_thres=self.genes[i].fitness
                self.fittest_index = i
        self.fittest_gene = self.fittest_index

        self.genes[self.fittest_index].replace = False 
        self.fittest_genes.append(self.genes[self.fittest_index])
        self.unfittest_genes.remove(self.genes[self.fittest_index])

#iterate several times for selecting top (1 - replacement_rate) percent genes to keep. 
        for i in range(int((1 - self.ga_type.replacement_rate)*self.population_size)):
            ftns_thres = np.inf
            best = 0
            for j in range(self.ga_type.tournament_size):
                cidx = int(np.random.random()*len(self.unfittest_genes))
                self.unfittest_genes[cidx]
                if self.unfittest_genes[cidx].fitness < ftns_thres:
                    best = cidx
                    ftns_thres = self.unfittest_genes[cidx].fitness
            self.unfittest_genes[best].replace=False
            self.fittest_genes.append(self.unfittest_genes[best])
            self.unfittest_genes.remove(self.unfittest_genes[best])

    def mutate(self):
        for i in self.genes:
           if i.replace:
               i = self.fittest_genes[int(np.random.random() * len(self.fittest_genes))].clone()
               i.mutate(self.ga_type.prob_mutation)

    def crossover(self):
        for c1 in self.genes:
            if c1.replace:
                if np.random.random() < self.ga_type.prob_crossover:
                    father = self.fittest_genes[int(np.random.random() * len(self.fittest_genes))]
                    mother = self.fittest_genes[int(np.random.random() * len(self.fittest_genes))]
                    sibling = self.unfittest_genes[int(np.random.random() * len(self.unfittest_genes))]

                    split = np.random.random() * self.no_of_assets

                    for j in range(self.no_of_assets):
                        sibling.chromosome[j] = mother.chromosome[j]
                        sibling.chromosome[j] = father.chromosome[j]
                        if j < split:
                            sibling.chromosome[j] = father.chromosome[j]
                            sibling.chromosome[j] = mother.chromosome[j]
                            


if __name__ == "__main__":
    df = pd.read_csv("stock_price.csv",index_col="date")
#    print(df)
    df = df.dropna()
    df = df.diff(1) / df.shift(1)
    df = df.dropna()
#    print(df)
    x = GeneticAlgorithm(df,ga_cycles = 5)
    x.run_softmax()
    print(x.best_portfolio("None"))






















