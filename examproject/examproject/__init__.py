#%%
# Importing all the necessary packages
import numpy as np
import sympy as sm
import scipy as sp
from scipy import optimize
from scipy import interpolate

import matplotlib.pyplot as plt
import ipywidgets as widgets
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
%matplotlib inline

# Problem 1:

# Define parameter values
rho = 2
beta = 0.96
gamma = 0.1
w = 2
b = 1
Delta = 0.1

# Define human capital vector 
h_vec = np.linspace(0.1, 1.5, 100) 

#%%
# Problem 1, Question 1:

# The basic functions are:

# Utility
def utility(c, rho):
    return c**(1-rho)/(1-rho)

# Consumption
def c(w, h, l, b):
    return w*h*l + b*(1-l)

# Disutility
def disutility(l, gamma):
    return gamma*l

# Total utility in period 2
def v2(w, h2, l2, b, rho, gamma):
    return utility(c(w, h2, l2, b), rho) - disutility(l2, gamma)

# Total utility in period 1
def v1(h1, l1, v2_interp, Delta, w, b, rho, gamma, beta):
    
    # From period 1 to period 2 the consumer can acummulate human capital based on the two functions below:
    
    # If human capital is not accumulated, utility in period 2 becomes
    N_h2 = h1 + l1 + 0
    N_v2 = v2_interp([N_h2])[0]
    
    # b. If human capital is accumulated, utility in period 2 becomes
    Y_h2 = h1 + l1 + Delta
    Y_v2 = v2_interp([Y_h2])[0]
    
    # c. Given the probabilities for ho
    v2 = 0.5*N_v2 + 0.5*Y_v2
    
    # d. total net utility
    return utility(c(w, h1, l1, b), rho) - disutility(l1, gamma) + beta*v2

#%%
# The solution function for period 2 is:

def solve_period_2(rho, gamma, Delta):
    
    # Create the vectors for period 2
    l2_vec = np.empty(100)
    v2_vec = np.empty(100)
    
    # Solve for each h2 in grid
    for i, h2 in enumerate(h_vec):
        
        # Choose either l2 = 0 or l2 = 1 by comparing the utility values from these two options
        if v2(w, h2, 1, b, rho, gamma) <= v2(w, h2, 0, b, rho, gamma):
            l2_vec[i] = 0
        else:
            l2_vec[i] = 1
        
        # Save the estimated values of v2, based on the choice of working or not 
        v2_vec[i] = v2(w, h2, l2_vec[i], b, rho, gamma)
        
    return l2_vec, v2_vec
    
# Solve for period 2
l2_vec, v2_vec = solve_period_2(rho, gamma, Delta)

# Figure
fig = plt.figure(figsize = (8,10))
ax = fig.add_subplot(2,1,1)
ax1 = fig.add_subplot(2,1,2)
ax.plot(h_vec, l2_vec)
ax1.plot(h_vec, v2_vec)


# Labels
ax.set_xlabel('Human Capital')
ax.set_ylabel('Labor Supply')
ax.set_title('Labor Supply and Human Capital- Period 2')

ax1.set_xlabel('Human Capital')
ax1.set_ylabel('Utility')
ax1.set_title('Utility and Human Capital- Period 2')

plt.tight_layout()

#%%
# Problem 1, Question 2:

# The solution function for period 1 is:

def solve_period_1(rho, gamma, beta, Delta, v1, v2_interp):
    
    # Vectors
    l1_vec = np.empty(100)
    v1_vec = np.empty(100)
    
    # Solve for each h1
    for i, h1 in enumerate(h_vec):
             
        # The individual decides whether to work or not by comparing his utility. If she is better off not working, we will have
        # l1=0 otherwise l1=1
        if v1(h1, 1, v2_interp, Delta, w, b, rho, gamma, beta) <= v1(h1, 0, v2_interp, Delta, w, b, rho, gamma, beta):
            l1_vec[i] = 0
        else:
            l1_vec[i] = 1
        
        v1_vec[i] = v1(h1, l1_vec[i], v2_interp, Delta, w, b, rho, gamma, beta)
        
    return l1_vec, v1_vec

# Construct interpolator
v2_interp = interpolate.RegularGridInterpolator((h_vec,), v2_vec, bounds_error=False, fill_value=None)

# Solve for period 1
l1_vec, v1_vec = solve_period_1(rho, gamma, beta, Delta, v1, v2_interp)

# Figure
fig = plt.figure(figsize = (8,10))
ax = fig.add_subplot(2,1,1)
ax1 = fig.add_subplot(2,1,2)
ax.plot(h_vec,l1_vec)
ax1.plot(h_vec, v1_vec)


# Labels
ax.set_xlabel('Human Capital')
ax.set_ylabel('Labor Supply')
ax.set_title('Labor Supply and Human Capital - Period 1')

ax1.set_xlabel('Human Capital')
ax1.set_ylabel('Utility')
ax1.set_title('Utility and Human Capital- Period 1')

plt.tight_layout()

#%%
# Problem 1, Question 3:

b = 2.2 # Set new value to the parameter of benefits

# Solve for period 2 and period 1
# We did not create new vectors, just use the old ones
l2_vec, v2_vec = solve_period_2(rho, gamma, Delta)
l1_vec, v1_vec = solve_period_1(rho, gamma, beta, Delta, v1, v2_interp)

# Figure
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(2,2,1)
ax1 = fig.add_subplot(2,2,2)
ax2 = fig.add_subplot(2,2,3)
ax3 = fig.add_subplot(2,2,4)

# Plots
ax.plot(h_vec,l2_vec)
ax1.plot(h_vec, l1_vec)
ax2.plot(h_vec, v2_vec)
ax3.plot(h_vec, v1_vec)

# Labels
ax.set_xlabel('Human Capital')
ax.set_ylabel('Labor Supply')
ax.set_title('Labor Supply and Human Capital - Period 2')

ax1.set_xlabel('Human Capital')
ax1.set_ylabel('Labor Supply')
ax1.set_title('Labor Supply and Human Capital - Period 1')

ax2.set_xlabel('Human Capital')
ax2.set_ylabel('Utility')
ax2.set_title('Utility and Human Capital- Period 2')

ax3.set_xlabel('Human Capital')
ax3.set_ylabel('Utility')
ax3.set_title('Utility and Human Capital- Period 1')

plt.tight_layout()

#%%
# Problem 2, Question 1:

# Define the parameter values
par = {}
par['alpha'] = 5.76
par['h'] = 0.5
par['b'] = 0.5
par['phi'] = 0
par['gamma'] = 0.075

# Activating pretty printing
sm.init_printing(use_unicode=True)

# Define all the needed variables for the symbolic solution
v = sm.symbols('v_t')
y_t = sm.symbols('y_t')  
pi_t = sm.symbols('pi_t') 
h = sm.symbols('h')
alpha = sm.symbols('alpha')
b = sm.symbols('b')

pi_t_1 = sm.symbols('\pi_{t-1}')
y_t_1 = sm.symbols('y_{t-1}')
s_t = sm.symbols('s_t')
s_t_1 = sm.symbols('s_{t-1}')
gamma = sm.symbols('gamma')
phi = sm.symbols('\phi')

# Check if they are printed properly
h, alpha, v, b, y_t, pi_t, pi_t_1, gamma, phi, y_t_1, s_t, s_t_1

AD = sm.Eq(pi_t, (1/(h*alpha))*(v - (1+b*alpha)*y_t))
SRAS = sm.Eq(pi_t, (pi_t_1 + gamma*y_t - phi*gamma*y_t_1 + s_t - phi*s_t_1))
AD, SRAS

#%%
# Solve for the equilibrium value of output

# Step 1
AD_1 = sm.solve(AD, pi_t)
AD_1

# Step 2
AD_2 = SRAS.subs(pi_t, AD_1[0])
AD_2

# Step 3
Output = sm.solve(AD_2, y_t)
Output

#Solving for the equilibrium value of inflation

# Step 1
AS_1 = SRAS.subs(y_t, Output[0])
AS_1

# Step 2
Inflation = sm.solve(AS_1, pi_t)
Inflation

# Deactivating pretty printing
sm.init_printing(use_unicode=False)

#%%
# Problem 2, Question 2:

# The two calculated equations are lambdified to be used
sol_output = sm.lambdify((y_t_1, s_t_1, pi_t_1, s_t, v, phi, alpha, gamma, h, b), Output[0])
sol_inflation = sm.lambdify((y_t_1, s_t_1, pi_t_1, s_t, v, phi, alpha, gamma, h, b), Inflation[0])

# Set the parameters equal to their values
def _sol_output(y_t_1, s_t_1, pi_t_1, s_t, v, phi=par['phi'], alpha=par['alpha'], gamma=par['gamma'], h=par['h'], b=par['b']):
    return sol_output(y_t_1, s_t_1, pi_t_1, s_t, v, phi, alpha, gamma, h, b)


def _sol_inflation(y_t_1, s_t_1, pi_t_1, s_t, v, phi=par['phi'], alpha=par['alpha'], gamma=par['gamma'], h=par['h'], b=par['b']):
    return sol_inflation(y_t_1, s_t_1, pi_t_1, s_t, v, phi, alpha, gamma, h, b)
                         

# The variables' values are inserted into the functions 
A = _sol_output(y_t_1=0, s_t_1=0, pi_t_1=0, s_t=0, v=0)
B = _sol_output(y_t_1=0, s_t_1=0, pi_t_1=0, s_t=0, v=0.1)

C = _sol_inflation(y_t_1=0, s_t_1=0, pi_t_1=0, s_t=0, v=0)
D = _sol_inflation(y_t_1=0, s_t_1=0, pi_t_1=0, s_t=0, v=0.1)
      
print('The values of y_t and pi_t when all the variables are equal to zero:')
print(f'y_t = {A}')
print(f'pi_t = {C}')
print('The values of y_t and pi_t when all the variables are equal to zero, except v = 0.1')
print(f'y_t = {B}')
print(f'pi_t = {D}')

#%%
# Define the AD and SRAS functions to plot them
def AD(h, alpha, v, b, y_t):
    return (1/(h*alpha))*(v - (1+b*alpha)*y_t)
    
def SRAS(pi_t_1, gamma, y_t, phi, y_t_1, s_t, s_t_1):
    return (pi_t_1 + gamma*y_t - phi*gamma*y_t_1 + s_t - phi*s_t_1)

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(1,1,1)

y_lin = np.linspace(-0.3, 0.3, 100)
AD_0 = AD(h=par['h'], alpha=par['alpha'], v=0, b=par['b'], y_t=y_lin)
AD_1 = AD(h=par['h'], alpha=par['alpha'], v=0.1, b=par['b'],y_t=y_lin)
SRAS_total = SRAS(pi_t_1=0, gamma=par['gamma'], y_t=y_lin, phi=par['phi'], y_t_1=0, s_t=0, s_t_1=0) 

plt.plot(y_lin, AD_0, label='AD 0 (with no disturbance)')
plt.plot(y_lin, AD_1, label='AD 1 (with demand disturbance, v=0.1)')
plt.plot(y_lin, SRAS_total, label='SRAS')
plt.plot(A, C, marker='.', color='black', label='Equilibrium before the demand disturbance')
plt.plot(B, D, marker='o', color='black', label='Equilibrium after the demand disturbance')
plt.grid(True)

plt.title('AD - SRAS')
plt.xlabel('$y_t$')
plt.ylabel('$\pi_t$')
plt.legend(loc='upper right')
plt.show()

#%%

#Problem 2, Question 3:

# Define more parameter values
par['delta'] = 0.80
par['omega'] = 0.15

def v_t(v_t_1, x_t, delta=par['delta']):
    return delta*v_t_1 + x_t

def s_t(s_t_1, c_t, omega=par['omega']):
    return omega*s_t_1 + c_t

# Create a random seed
seed = 2019
np.random.seed(seed)

# Define four periods to check which one will present the evolution of the economy better
#T1 = 25
#T2 = 50
#T3 = 100
T4 = 125

# Creating the vectors that will be needed for the simulation
y_vec = [0]
pi_vec = [0]
v_vec = [0]
x_vec = np.zeros(T4) # all the demand shocks are set equal to 0
x_vec[1] = 0.1 # the second element of the demand shocks list is set to 0.1, because demand disturbance function uses v_{t-1}
s_vec = [0]
c_vec = np.zeros(T4) # all the supply shocks are 0 

# Creating a for loop to fill in the vectors
for t in range (1, T4):
    v_vec.append(v_t (v_vec [t-1], x_vec[t]))
    s_vec.append(s_t (s_vec [t-1], c_vec[t]))
    y_vec.append(sol_output (y_vec[t-1], s_vec[t-1], pi_vec[t-1], s_vec[t], v_vec[t], par['phi'], par['alpha'], par['gamma'], par['h'], par['b']))
    pi_vec.append(sol_inflation (y_vec[t-1], s_vec[t-1], pi_vec[t-1], s_vec[t], v_vec[t], par['phi'], par['alpha'], par['gamma'], par['h'], par['b']))

# Checking the created vectors (uncomment next row to check the values from any of the vectors) 
#y_vec, pi_vec, v_vec, s_vec

periods = np.linspace(0, T4, T4)

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(1,1,1)

ax.plot(periods, pi_vec, label='$\pi_t$')
ax.plot(periods, y_vec, label='$y_t$')
plt.grid(True)

plt.ylim(-0.005, 0.025)
ax.set_title('Evolution of output and inflation')
plt.xlabel('$Periods$')
plt.ylabel('$y_t$ and $\pi_t$')
plt.legend(loc='upper right')

#%%
#Problem 2, Question 4:

# Define more parameter values
par['sigma_x'] = 3.492
par['sigma_c'] = 0.2

# Create a random seed
seed = 2020
np.random.seed(seed)

# Define the periods of the model simulation
T5 = 1000

# Creating the vectors that will be needed for the simulation
y_vec_1 = [0]
pi_vec_1 = [0]
v_vec_1 = [0]
x_vec_1 = np.random.normal(loc=0, scale=par['sigma_x'], size=T5) # all the demand shocks are normally distributed
s_vec_1 = [0]
c_vec_1 = np.random.normal(loc=0, scale=par['sigma_c'], size=T5) # all the supply shocks are normally distributed 

# Creating a for loop to fill in the vectors
for t in range (1, T5):
    v_vec_1.append(v_t (v_vec_1[t-1], x_vec_1[t]))
    s_vec_1.append(s_t (s_vec_1[t-1], c_vec_1[t]))
    y_vec_1.append(sol_output (y_vec_1[t-1], s_vec_1[t-1], pi_vec_1[t-1], s_vec_1[t], v_vec_1[t], par['phi'], par['alpha'], par['gamma'], par['h'], par['b']))
    pi_vec_1.append(sol_inflation (y_vec_1[t-1], s_vec_1[t-1], pi_vec_1[t-1], s_vec_1[t], v_vec_1[t], par['phi'], par['alpha'], par['gamma'], par['h'], par['b']))

periods = np.linspace(0, T5, T5)

fig = plt.figure(figsize = (15,6))
fig.suptitle('Graphical simulation of the model', fontsize=20)

ax1 = fig.add_subplot(2,1,1)
ax1.plot(periods, pi_vec_1, label='$\pi_t$')
ax1.set_title('Evolution of Inflation')
plt.ylabel('$\pi_t$')
plt.grid(True)
plt.legend(loc='upper right')

ax2 = fig.add_subplot(2,1,2)
ax2.plot(periods, y_vec_1, color='orange', label='$y_t$')
ax2.set_title('Evolution of Output')
plt.ylabel('$y_t$')
plt.xlabel('$Periods$')
plt.grid(True)
plt.legend(loc='upper right')

#%%
# Calculate the requested statistics:
var_y_t = np.var(y_vec_1)
var_pi_t = np.var(pi_vec_1)
corr_y_t_pi_t = np.corrcoef(y_vec_1, pi_vec_1)
corr_y_t_y_t_1 = np.corrcoef(y_vec_1[1:], y_vec_1[:-1])
corr_pi_t_pi_t_1 = np.corrcoef(pi_vec_1[1:], pi_vec_1[:-1])

# Print the statistics
print(f'The variance of the output is: var(y_t) = {var_y_t}')
print(f'The variance of the inflation is: var(pi_t) = {var_pi_t}')
print(f'The correlation between the inflation and the output is: corr(pi_t, y_t) = {corr_y_t_pi_t[0,1]}')
print(f'The auto-correlation between the output (of the current period) and the output (of the previous period) is: corr(y_t,y_t_1) = {corr_y_t_y_t_1[0,1]}')
print(f'The auto-correlation between the inflation (of the current period) and the inflation (of the previous period) is: corr(pi_t, pi_t_1) = {corr_pi_t_pi_t_1[0,1]}')

#%%
# Problem 2, Question 5:

# Create a random seed
seed = 2021
np.random.seed(seed)

# Define the periods of the model simulation
T6 = 1000

# Define the demand and supply shocks as normal distributions
x_vec_2 = np.random.normal(loc=0, scale=par['sigma_x'], size=T6)
c_vec_2 = np.random.normal(loc=0, scale=par['sigma_c'], size=T6)

# Create a function that will fill in the vectors and the only function parameter will be phi

def phi_function(phi):
    
    # Create vectors
    y_vec_2 = [0]
    pi_vec_2 = [0]
    v_vec_2 = [0]
    s_vec_2 = [0]
    corr_y_t_pi_t_vec = [0]
    
    # In (y_vec_2.append...) and (pi_vec_2.append...) all the parameters are substituted with their values apart from phi,
    # which will take the value that we will give when calling the function.
    for t in range (1, T6):
        v_vec_2.append(v_t (v_vec_2[t-1], x_vec_2[t]))
        s_vec_2.append(s_t (s_vec_2[t-1], c_vec_2[t]))
        y_vec_2.append(sol_output (y_vec_2[t-1], s_vec_2[t-1], pi_vec_2[t-1], s_vec_2[t], v_vec_2[t], phi, par['alpha'], par['gamma'], par['h'], par['b']))
        pi_vec_2.append(sol_inflation (y_vec_2[t-1], s_vec_2[t-1], pi_vec_2[t-1], s_vec_2[t], v_vec_2[t], phi, par['alpha'], par['gamma'], par['h'], par['b']))
        
    corr_y_t_pi_t_vec = np.corrcoef(y_vec_2, pi_vec_2)[1,0]
    
    return y_vec_2, pi_vec_2, corr_y_t_pi_t_vec

# Unpack solution
y_vec_2, pi_vec_2, corr_y_t_pi_t_vec = phi_function(par['phi'])

# Simulate the model and fill in the correlation vector

phi_vec = np.linspace(1e-8, 9.9999999e-1, T6) # define different values for phi between 0 and 1.
# At the beggining we used 0.00000001 and 0.99999999 as the bound for phi values, then print phi_vec 
# and finally used the first and last printed value as bounds

correlations = []

for i in phi_vec:
    y_vec_2, pi_vec_2, corr_y_t_pi_t_vec = phi_function(i)
    correlations.append(corr_y_t_pi_t_vec)

# Plot the correlations and the respective phi values

plt.figure(figsize = (12,8))
plt.plot(phi_vec, correlations, label='Correlation ($y_{t},\pi_{t}$)')
plt.axhline(y=0.31, linestyle='--', color='black', xmin=0, label='Correlation ($y_{t},\pi_{t}$) = 0.31')

plt.title('Correlations of ($y_{t},\pi_{t}$) for different values of $\phi$')
plt.ylabel('Correlations ($y_{t},\pi_{t}$)')
plt.xlabel('$\phi$')
plt.ylim(-0.2, 0.5)
plt.grid(True)
plt.legend(loc='upper left')

#%%
# Define the function that will be used in the optimizer
obj = lambda phi: (np.corrcoef(phi_function(phi)[0], phi_function(phi)[1])[1,0] - 0.31)

# Use two optimizing methods to estimate and cross-check the result
brentq_result = optimize.brentq(obj, a=0, b=1, full_output=False)
print(f'The value of phi estimated with the brentq numerical optimizer is: brentq_result = {brentq_result}')
bisect_result = optimize.bisect(obj, a=0, b=1, full_output=False)
print(f'The value of phi estimated with the bisect numerical optimizer is: bisect_result = {bisect_result}')

#%%
# Problem 2, Question 6:

#%%
# Problem 3:

# a. parameters
N = 50000
mu = np.array([3,2,1])
Sigma = np.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]])
gamma = 0.8
zeta = 1

# b. random draws
seed = 1986
np.random.seed(seed)

# preferences
alphas = np.exp(np.random.multivariate_normal(mu, Sigma, size=N))
betas = alphas/np.reshape(np.sum(alphas,axis=1), (N,1))

# endowments
e1 = np.random.exponential(zeta,size=N)
e2 = np.random.exponential(zeta,size=N)
e3 = np.random.exponential(zeta,size=N)

#%%
# Problem 3, Question 1:

# Plot the histogram of the budgets share for each good 
fig = plt.figure(dpi=100)
ax = fig.add_subplot(1,1,1)

goods = ['Good 1', 'Good 2', 'Good 3']
ax.hist(betas, bins=50, label=goods)

ax.set_title('The budget shares of three goods')
ax.set_xlabel('betas')
ax.set_ylabel('Consumers')
plt.legend(loc='upper right')

#%%
#Problem 3, Question 2:

# Demand function:

def demand_good_1_fun(betas, p1, p2, e1, e2, e3):
    I = p1*e1 + p2*e2 + e3
    return betas[:,0]*I/p1

def demand_good_2_fun(betas, p1, p2, e1, e2, e3):
    I = p1*e1 + p2*e2 + e3
    return betas[:,1]*I/p2

def demand_good_3_fun(betas, p1, p2, e1, e2, e3):
    I = p1*e1 + p2*e2 + e3
    return betas[:,2]*I

# Excess demand function:

def excess_demand_good_1_func(betas, p1, p2, e1, e2, e3):
    
    demand = np.sum(demand_good_1_fun(betas, p1, p2, e1, e2, e3))
    supply = np.sum(e1)
    excess_demand = demand - supply
    
    return excess_demand

def excess_demand_good_2_func(betas, p1, p2, e1, e2, e3):

    demand = np.sum(demand_good_2_fun(betas, p1, p2, e1, e2, e3))
    supply = np.sum(e2)
    excess_demand = demand - supply
    
    return excess_demand

#%%
# Return coordinate matrices from coordinate vectors
p1_vec = np.linspace(1, 10, 100)
p2_vec = np.linspace(1, 10, 100)
p1_grid, p2_grid = np.meshgrid(p1_vec, p2_vec) 

# Store the function value in two-dimention lists
excess_1_grid = np.ones((100, 100))
excess_2_grid = np.ones((100, 100))

for i, p1 in enumerate(p1_vec):
    for j, p2 in enumerate(p2_vec):
        excess_1_grid[i,j] = excess_demand_good_1_func(betas, p1, p2, e1, e2, e3)
        excess_2_grid[i,j] = excess_demand_good_2_func(betas, p1, p2, e1, e2, e3)

# Plot the excess demand function for three goods
fig = plt.figure(figsize = (10,15))

# Excess demand for good 1
ex1 = fig.add_subplot(2,1,1, projection='3d')
ex1.plot_surface(p1_grid, p2_grid, excess_1_grid)
ex1.invert_xaxis()

ex1.set_xlabel('$p_1$')
ex1.set_ylabel('$p_2$')
ex1.set_zlabel('Excess Demand')
ex1.set_title('Good 1')

# Excess demand for good 2
ex2 = fig.add_subplot(2,1,2, projection='3d')
ex2.plot_surface(p1_grid, p2_grid, excess_2_grid)
ex2.invert_xaxis()

ex2.set_xlabel('$p_1$')
ex2.set_ylabel('$p_2$')
ex2.set_zlabel('Excess Demand')
ex2.set_title('Good 2')

plt.tight_layout()

#%%
# Problem 3, Question 3:

# Create a function to find the equilibrium prices 
def find_equilibrium(betas, p1, p2, e1, e2, e3, kappa=0.5, eps=1e-5, maxiter=5000):
    
    t = 0
    while True:

        # a. step 1: excess demand
        Z1 = excess_demand_good_1_func(betas, p1, p2, e1, e2, e3)
        Z2 = excess_demand_good_2_func(betas, p1, p2, e1, e2, e3)
        
        # b: step 2: stop?
        if  np.abs(Z1) < eps and np.abs(Z2) < eps or t >= maxiter:
            print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand 1 -> {Z1:14.8f}')
            print(f'{t:3d}: p2 = {p2:12.8f} -> excess demand 2 -> {Z2:14.8f}')
            break    
    
        # c. step 3: update p1
        p1 = p1 + kappa*Z1/N
        p2 = p2 + kappa*Z2/N
            
        # d. step 4: return 
        if t < 5 or t%250 == 0:
            print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand 1 -> {Z1:14.8f}')
            print(f'{t:3d}: p2 = {p2:12.8f} -> excess demand 2 -> {Z2:14.8f}')
        elif t == 5:
            print('   ...')
            
        t += 1    

    return p1, p2

# Find the equilibrium prices
p1 = 1.5
p2 = 2
kappa = 0.5
eps = 1e-5

p1, p2 = find_equilibrium(betas, p1, p2, e1, e2, e3, kappa=kappa, eps=eps)

# Ensure that excess demand of both goods are (almost) zero
Z1 = excess_demand_good_1_func(betas, p1, p2, e1, e2, e3)
Z2 = excess_demand_good_2_func(betas, p1, p2, e1, e2, e3)
print(Z1, Z2)
assert(np.abs(Z1) < eps)
assert(np.abs(Z2) < eps)

#%%
# Problem 3, Question 4:

# Use the price value in equilibrium 
p1 = 6.4900
p2 = 2.6166

# Calculate the utility function
def utility(betas, e1, e2, e3, gamma):
    
    I = 6.49*e1 + 2.62*e2 + e3
    x1 = betas[:,0]*(I/6.49)
    x2 = betas[:,1]*(I/2.62)
    x3 = betas[:,2]*I
    
    return (x1**betas[:,0] + x2**betas[:,1] + x3**betas[:,2])**gamma

# Plot the utility function
U = utility(betas,e1, e2, e3, gamma)

plt.hist(U,100)
plt.xlabel('Utility')
plt.ylabel('Consumers')
plt.title('Utilities Distribution In the Walras-equilibrium ')

mean = np.mean(U)
variance = np.var(U)

print(f'mean = {mean:.3f}, variance = {variance:.3f}')

#%%
# Problem 3, Question 5:

# Create equally distributed endowments
e_1 = np.mean(e1) + np.zeros(N)
e_2 = np.mean(e2) + np.zeros(N)
e_3 = np.mean(e3) + np.zeros(N)

# Find the equilibrium prices
p1 = 6
p2 = 2
p1, p2 = find_equilibrium(betas, p1, p2, e_1, e_2, e_3, kappa=kappa, eps=eps)

#%%
# New equilibrium prices
p1 = 6.4860
p2 = 2.6172

# Define the new utility function when endowments changed
def utility_1(betas,e_1, e_2, e_3, gamma):
    
    I = 6.4860*e_1 + 2.6172*e_2 + e_3
    x1 = betas[:,0]*(I/6.4860)
    x2 = betas[:,1]*(I/2.6172)
    x3 = betas[:,2]*(I/1)
    
    return (x1**betas[:,0] + x2**betas[:,1] + x3**betas[:,2])**gamma

U_1 = utility_1(betas, e_1, e_2, e_3, gamma)

# Plot the original and new utility function
plt.hist(U, 100, label='Original Utility Distribution')
plt.hist(U_1, 100, label='New Utility Distribution')
plt.xlabel('Utility')
plt.ylabel('Consumers')
plt.title('Utilities Distribution')
plt.legend()

mean_u = np.mean(U)
var_u = np.var(U)

mean_u_1 = np.mean(U_1)
var_u_1 = np.var(U_1)

print(f'original utilty: mean = {mean_u:.3f}, variance = {var_u:.3f}')
print(f'new utilty     : mean = {mean_u_1:.3f}, variance = {var_u_1:.3f}')

#%%
# Plot the original and new utility function in one graph to compare the difference
def interactive(gam):
    uti = utility(betas, e_1, e_2, e_3, gamma=gam)
    uti_1 = utility_1(betas,e_1, e_2, e_3, gamma=gam)
    
    plt.hist(uti, 500, label='Original Utility Distribution')
    plt.hist(uti_1, 500, label='New Utility Distribution')
    plt.xlabel('Utility')
    plt.ylabel('Consumers')
    plt.title('Utilities Distribution')
    plt.legend()
     
widgets.interact(interactive,
    gam = widgets.FloatSlider(description='$\\gamma$', min=-1, max=1, step=0.1, value=0)
