"""
    CS 4180/5180 RL and SDM
    Exercise 3: Dynamic Programming
    Prof: Robert Platt
    Date: October 9th, 2021
    Author: Guanang Su
"""

import numpy as np

def iterativePolicyEvaluation(policy, theta, V, state, next_state, reward):

    while True:
        delta = 0
        for s in state:
            v = V(s)
            tri = max()

        if delta < theta:
            break

    return np.around(V, 1)



def valueIteration(policy, threshold, V, state, next_state, reward):
    tri = 0
    while tri < threshold:
        tri = 0
        for s in state:
            v = V(s)
            tri = max()
    pi_s = np.argmax()


def policyIteration(p, the, V, s, ns, r):
    iterativePolicyEvaluation(p, the, V, s, ns, r)
    policy_stable = False
    if policy_stable == True:


