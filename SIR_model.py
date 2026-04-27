"""
SIR Model developed in class
"""
from matplotlib import pyplot as plt 
import math
N= 100 # Number of people in the population
I, R = 1, 0  #initialization of infected (I) and recovered (R) populations
S = N - I  #initialization of suseptible population
beta, gamma = 0.3, 0.1 
time = list(range(0,999)) #initialization of time (days)
S_list = []
I_list = []
R_list = []
    
_lambda = -(math.log(.5)/270) #immunity after 9 months 
V = 0.175 * (1/365) # vaccination rate
for d in time: # Main loop
    dS = -beta * S * I / N              #Calcs
    dI = beta * S * I / N - gamma * I   # |
    dR = gamma * I                      # V

    S += dS
    I += dI 
    R += dR

    S_list.append(S)
    I_list.append(I)
    R_list.append(R)


plt.plot(S_list,'orange', label='Susceptible')
plt.plot(I_list,'r', label='Infected')
plt.plot(R_list,'g', label='Recovered with immunity')
plt.xlabel('Time t, [days]')
plt.ylabel('Numbers of individuals')
plt.legend()
plt.show()

"""
Simple simulations can be extremely useful. Keep em simple.
"""