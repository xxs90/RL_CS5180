#!/usr/bin/python

import numpy as np
from scipy.stats import poisson


# Here is my implementation of the open_to_close function, which calculates
# the transition dynamics, P, and the reward function, R
# We need to account for all possible numbers of cars that could be rented out, and 
# cars that could be returned, to know what our next state will be. 
def open_to_close(P, R, lambda_requests, lambda_dropoffs):
    # Set up a requests and request probability variable.
    requests = 0
    request_prob = poisson(lambda_requests).pmf(requests)
    # Once we get to a probability below theta, it's so small we can ignore.
    while request_prob > theta:
        # for each possible number of starting cars (NOTE: we can have up to 25 (20 + send over 5)).
        for n in range(26):
            # Increase the reward by 10 * the probability of that reward * the number rented out
            R[n] += (10 * request_prob * min(requests, n))
        # Figure out how many cars were returned.
        dropoffs = 0
        drop_prob = poisson(lambda_dropoffs).pmf(dropoffs)
        # end the loop once our return probability is very small.
        while drop_prob > theta:
            # Remember we can have up to 25 cars.
            for n in range(26):
                satisfied_requests = min(requests, n)
                # can't have more than 20, or less than 0, cars at the end of the day.
                new_n = max(0, min(20, (n + dropoffs) - satisfied_requests))
                # Increment the dynamics probability by the probability we get that many rentals and
                # that many dropoffs
                P[n][new_n] += request_prob * drop_prob
            # increment dropoffs, recalculate the probability of getting that many dropoffs
            dropoffs += 1
            drop_prob = poisson(lambda_dropoffs).pmf(dropoffs)
        # Increment requests, and recalculate the probability of getting that many requests.
        requests += 1
        request_prob = poisson(lambda_requests).pmf(requests)


# Here is the function to get the estimated reward for an action
# We need to sum the probabilities over all possibilities given the 
# current state that we end up in the next state, and and rewards for
# getting to that state. This also includes the discounted value of the
# next state we would be in.
def get_estimated_return(x, y, a):
    a = max(-y, min(a, x))
    a = max(-5, min(5, a))
    morning_x = int(x - a)
    morning_y = int(y + a)
    # It costs $2 per car moved.
    val = -2 * abs(a)
    for new_x in range(21):
        for new_y in range(21):
            val += P_loc1[morning_x][new_x] * P_loc2[morning_y][new_y] * (R_loc1[morning_x] + R_loc2[morning_y] +
                                                                     gamma * V[new_x][new_y])
    return val


# Standard policy evaluation implementation.
# Loop over all possible states, take a 1-step lookahead
# to get the estimated value for that state plus action pair, 
# and then evaluate how good the policy is.
def iterative_policy_evaluation():
    while True:
        delta = 0
        for x in range(21):
            for y in range(21):
                old_v = V[x][y]
                a = pi[x][y]
                V[x][y] = get_estimated_return(x, y, a)
                new_delta = abs(old_v - V[x][y])
                if new_delta > delta:
                    delta = new_delta

        if delta < theta:
            break


# This policy function takes in an state (x, y), and output the
# best action based on our estimated return.
def policy(x, y, epsilon=0.0000000001):
    best_value = -1
    best_action = None
    for a in range(max(-5, -y), min(5, x)+1):
        this_value = get_estimated_return(x, y, a)
        if this_value > (best_value + epsilon):
            best_value = this_value
            best_action = a
    return best_action


# Function to improve the policy.
# for all possible states, if our current policy's action is 
# worse than what we estimate to be the best possible action,
# change the action. As long as we make at least one change,
# then our policy has been improved.
def policy_improvement():
    policy_improved = False
    for x in range(21):
        for y in range(21):
            b = pi[x][y]
            pi[x][y] = policy(x, y)
            if b != pi[x][y]:
                policy_improved = True
    show_policy()
    return policy_improved


# Function to print out the policy, and format it in a way for positive and negative
# numbers to print well.
def show_policy():
    for x in range(21):
        for y in range(21):
            if policy(x, y) < 0:
                print("{} ".format(policy(x, y)), end='')
            else:
                print(" {} ".format(policy(x, y)), end='')
        print()


# Policy iteration. I save the policy pi at each iteration (for plotting),
# perform iterative policy evaluation, then improve the policy. If the policy
# cannot be improved anymore, we are done.
def policy_iteration():
    count = 0
    while True:
        np.save("pi_{}".format(count), pi)
        iterative_policy_evaluation()
        count += 1
        print(count)
        if not policy_improvement():
            break


# Set up global variables to be used throughout the program, 
# Like the value function, V, the policy, pi, the
# transition and reward functions for both locations 1 and 2,
# gamma, the discounting factor, and theta, for accuracy calculation.
if __name__ == '__main__':
    V = np.zeros((21, 21), dtype=np.float32)
    rental_lambda_loc1 = 3
    rental_lambda_loc2 = 4
    return_lambda_loc1 = 3
    return_lambda_loc2 = 2
    pi = np.zeros((21, 21), dtype=int)
    P_loc1 = np.zeros((26, 21), dtype=np.float32)
    P_loc2 = np.zeros((26, 21), dtype=np.float32)
    R_loc1 = np.zeros(26)
    R_loc2 = np.zeros(26)
    gamma = 0.9
    theta = 0.0000001

    open_to_close(P_loc1, R_loc1, rental_lambda_loc1, return_lambda_loc1)
    open_to_close(P_loc2, R_loc2, rental_lambda_loc2, return_lambda_loc2)

    policy_iteration()
    np.save("problem6a_plotting/V", V)
