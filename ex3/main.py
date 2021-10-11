"""
    CS 4180/5180 RL and SDM
    Exercise 3: Dynamic Programming
    Prof: Robert Platt
    Date: October 9th, 2021
    Author: Guanang Su
"""
import numpy as np
import env
import algorithms as ag

def gridWorld():
    V = np.zeros(shape=(5, 5), dtype=float)
    S = np.empty(shape=(5, 5), dtype=str)
    V = ag.iterativePolicyEvaluation(V, S, 0.00001, 0.9)
    print(V)

#def carRental():


def main():
    str = input("Which problem do you choose? 'g': gridWorld, 'c': jackCarRental")
    gridWorld()
    #carRental()


if __name__ == "__main__":
    main()