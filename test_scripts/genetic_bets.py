'''
There's something wrong with this...
We're simulating winnings against the odds set...
the expected profit of this will always be zero.
We need to calculate our own odds and simulate against the sportsbook odds.
'''

import random
import numpy as np

def genetic_bets(A=[-300, -200, 100, 200, 1400], B=100,
                pop_size=50, gens = 50, sim_iter=2000,
                P_c=0.25, P_m=0.1):

    P = ao_to_percentage(A)
    Z = ao_to_earnings(A)
    Y = generate_population(A, B, pop_size)

    for i in np.arange(gens):

        # Add Next Generation
        children = generate_children(Y, P_c, P_m, len(A)-1)
        Y = Y + children

        # Phase Out Previous Generation
        Y, avg_profit = natural_selection(Y, P, Z, B, sim_iter)
        np.random.shuffle(Y)

        # Print to console
        print(f'Generation {i}: {avg_profit} Avg Profit ----')

    # Develop a betting strategy base on the last generation
    S = betting_strategy(Y, P, Z, sim_iter)
    print(S)

    # Quick math best bet
    BB = B*P/np.sum(P)
    print(BB)


# Convert American Odds to Percentage
def ao_to_percentage(A):
    pp = []
    for aa in A:
        if aa < 0:
            aa = abs(aa)
            pp.append(aa/(aa+100))
        elif aa > 0:
            pp.append(100/(aa+100))

    return pp


# Convert American Odds to Earnings
def ao_to_earnings(A):

    Z = []
    for aa in A:
        if aa < 0:
            aa = abs(aa)
            Z.append(100/aa)
        elif aa > 0:
            Z.append(aa/100)

    return Z


# Generate an initial population
def generate_population(A, B, pop_size):

    Y = []
    ness = len(A)-1

    for ii in np.arange(pop_size):
        y = np.zeros(B)
        for jj in np.arange(B):
            y[jj] = int(random.randint(0,ness))

        Y.append(y.tolist())

    return Y


# Generate Children for GA
def generate_children(Y, P_c, P_m, ness):

    # Dimensionality
    n = len(Y)
    m = len(Y[0])

    # Create children via crossover
    children = []
    for i in range(0,n,2):
        parent1 = Y[i]
        parent2 = Y[i+1]
        for j in np.arange(m):
            if np.random.binomial(1, P_c):
                allele1 = parent1[j]
                allele2 = parent1 [j]

                parent1[j] = allele2
                parent2[j] = allele1

        children.append(parent1)
        children.append(parent2)

    # Handle Mutatations
    for i in np.arange(len(children)):
        for j in np.arange(len(children[i])):
            if np.random.binomial(1, P_m):
                children[i][j] = random.randint(0,ness)

    return children


# Select surviving individuals
def natural_selection(Y, P, Z, B, sim_iter):

    average_profits = []
    for y in Y:
        average_profits.append(simulate_earnings(y, P, Z, sim_iter) - B)

    sorted_Y = [y for _,y in sorted(zip(average_profits,Y))]
    new_gen = sorted_Y[int(len(Y)/2):int(len(Y))]
    return new_gen, np.mean(average_profits)


# Get a betting strategy
def betting_strategy(Y, P, Z, B, sim_iter):

    average_profits = []
    for y in Y:
        average_profits.append(simulate_earnings(y, P, Z, sim_iter) - B)

    sorted_Y = [y for _,y in sorted(zip(average_profits,Y))]
    best_Y = sorted_Y[int(len(Y))]

    strategy = np.zeros(len(P))
    for yy in best_Y:
        strategry[yy] = strategry[yy] + 1

    return strategy


# Simulate betting profits
def simulate_earnings(y, P, Z, sim_iter):

    earnings = []
    for i in np.arange(sim_iter):
        ee = 0
        for j in np.arange(len(P)):
            y_j = float(np.sum(np.array(y) == j))
            if np.random.binomial(1, P[j]):
                ee = ee + y_j + y_j*Z[j]

        earnings.append(ee)

    return sum(earnings)/sim_iter


if __name__ == "__main__":
    genetic_bets()
