#The OLG Model
#%%
# 1. Setup

#importing the necessary packages 
import numpy as np
import sympy as sm
import matplotlib.pyplot as plt
import ipywidgets as widgets

# 2. Symbolic solution of the Household's (i) and Firm's (ii) problems
# (i) Household's problem

#%%
# Activating pretty printing
sm.init_printing(use_unicode=True)

# Defining symbolically the necessary variables
c1 = sm.symbols('c_1_t')
c2 = sm.symbols('c_2_t+1')
beta = sm.symbols('beta')
z = sm.symbols('z_t+1') 
w = sm.symbols('w_t') # wage
r1 = sm.symbols('r_t+1')
u = sm.symbols('U_t') #Utility
L = sm.symbols('L') #Langrange
omega = sm.symbols('omega') #The Langrangian multiplier

# Checking that our variables have the proper format
c1,c2, beta, w, z, r1, L, omega

#%%
# Defining the utility function
def utility(c1, c2, beta):
    return (sm.log(c1) + beta*sm.log(c2))

#%%
#Defining the two budget constraints symbolically
budget_constraint_1 = sm.Eq(z, w - c1)
budget_constraint_2 = sm.Eq(c2, z*(1+r1))
budget_constraint_1, budget_constraint_2

#%%
#Solve the two budget constraints to obtain the Lifetime Budget Constraint in three steps:
# 1.Solve  for 𝑧𝑡+1  from the budget constraint 1
# 2.Substitute in budget constraint 2
# 3.Solve for  𝑤𝑡

# Step 1
z_from_constraint = sm.solve(budget_constraint_1, z)
z_from_constraint

#%%
# Step 2
z_subs_constraint = budget_constraint_2.subs(z, z_from_constraint[0])
z_subs_constraint

#%%
# Step 3
lifetime_budget_constraint = sm.solve(z_subs_constraint, w)
lifetime_budget_constraint

#%%
# Defining the Lifetime Budget Constraint symbolically
budget_constraint = w - c1 - c2/(1+r1)
budget_constraint

#%%
#Constructing the Langrangian
Langrangian = utility(c1, c2, beta) + omega * budget_constraint
Langrangian

#%%
#Solve in six steps:
# 1.Calculate the two FOCs wrt.  𝑐1,𝑡  and  𝑐2,𝑡+1 
# 2.Isolate the first FOC for  𝜔 
# 3.Substitute in the second FOC
# 4.Isolate the second FOC for  𝑐1,𝑡 
# 5.Substitute into the Lifetime Budget Constraint
# 6.Solve for  𝑐2,𝑡+1  to find the consumption in period 2
# 7.Substitute  𝑐2,𝑡+1  in Lifetime Budget Constraint
# 8.Solve for  𝑐1,𝑡  to find the consumption in period 1
# 9.Substitute in the first budget constraint ( 𝑧𝑡+1=𝑤𝑡−𝑐1,𝑡 )
# 10.Solve for  𝑧𝑡+1  to find the assets held by the household at period 2

# Step 1
foc1 = sm.diff(Langrangian, c1)
foc2 = sm.diff(Langrangian, c2)
foc1, foc2

#%%
# Step 2
foc_1 = sm.solve(foc1, omega)
foc_1

#%%
#Step 3
foc_2 = foc2.subs(omega, foc_1[0])
foc_2

#%%
# Step 4
AB = sm.solve(foc_2, c1)
AB

#%%
# Step 5
ABC = budget_constraint.subs(c1, AB[0])
ABC

#%%
# Step 6
consumption_2 = sm.solve(ABC,c2)
consumption_2

#%%
# Step 7
ABCD = budget_constraint.subs(c2, consumption_2[0])
ABCD

#%%
# Step 8
consumption_1 = sm.solve(ABCD, c1)
consumption_1

#%%
# Step 9
ABCDE = budget_constraint_1.subs(c1, consumption_1[0])
ABCDE

#%%
# Step 10
Assets = sm.solve(ABCDE, z)
Assets

# (ii) Firm's Problem

#%%
# Defining symbolically the necessary variables
alpha = sm.symbols('alpha') # share of physical capital
k = sm.symbols ('k_t')  # capital per effective worker
r = sm.symbols('r_t') # interest rate
delta = sm.symbols('delta') # depreciation rate of capital per period
A = sm.symbols('A_t') # technological index at period t
A1 = sm.symbols('A_{t+1}') 
g = sm.symbols ('g') #rate of technological growth
L0 = sm.symbols('L_{t}')
L1 = sm.symbols('L_{t+1}')
n = sm.symbols('n') #growth of labor force

# Checking that our variables have the proper format
alpha, k, r, delta, A, A1, g, L0, L1, n

#%%
tech = sm.Eq((1+g)*A, A1)
labor = sm.Eq((1+n)*L0, L1)
tech, labor

#%%
tech0 = sm.solve(tech, A)
labor0 = sm.solve(labor, L0)
tech0, labor0

#%%
# Defining the two FOC's and setting them equal with their rental rates
MPK = sm.Eq((alpha*k**(alpha-1)), r+delta)
MPL = sm.Eq (((1-alpha)*A*k**alpha), w)
MPK, MPL

#%%
Wage = sm.solve(MPL, w)
Wage

# 3. Symbolic (iii) and numerical (iv) solution for the aggregate level of the model
# (iii) Symbolic solution for the aggregate level of the model

#%%
# Defining symbolically the necessary variables
Ks = sm.symbols('K_{t+1}^{s}') #aggregate supply of physical capital at period t+1
Kd = sm.symbols('K_{t+1}^{s}')#aggregate demand of physical capital at period t+1
K = sm.symbols('K_{t+1}')
g1 = sm.symbols('g^{*}')
k1 = sm.symbols('k_{t+1}')

# Checking that our variables have the proper format
Ks, Kd, K, g1, k1

#%%
#Solve in seven steps:
# 1.Define the capital supply  𝐾𝑠𝑡+1 
# 2.Substitute for  𝑧𝑡+1  as calculated in the Household's problem
# 3.Substitute for  𝑤𝑡  as calculated in the Firm's problem
# 4.Substitute  𝐴𝑡  with  𝐴𝑡+1 
# 5.Substitute  𝐿𝑡  with  𝐿𝑡+1 
# 6.Substitute  (1+𝑛)(1+𝑔)  with  𝑔∗ 
# 7.Equalize  𝐾𝑠𝑡+1=𝐾𝑑𝑡+1=𝐾𝑡+1

# Step 1
capital_supply = sm.Eq(L0*z ,Ks)
capital_supply

#%% 
# Step 2
H = capital_supply.subs(z, Assets[0])
H

#%%
# Step 3
I = H.subs(w, Wage[0])
I

#%%
# Step 4
J = I.subs(A, tech0[0])
J

#%%
# Step 5
M = J.subs(L0, labor0[0])
M

#%%
# Step 6
N = M.subs(((1+g)*(1+n)), (g1+1))
N

#%%
# Step 7
P = N.subs(Ks, K)
P

#%%
sm.init_printing(use_unicode=False)

#(iv) Numerical solution for the aggregate level of the model

#%%
# Converting the calculated equations to python function
g1 = lambda n, g: (1+g)*(1+n)
solution = lambda k, alpha = 0.4 , g = 0.01, n = 0.06, beta = 0.8: (1-alpha)*k**alpha / ((1+g1(n,g))*(1+1/beta))
y = lambda x: x # this will be used to plot a 45o degree line at the first plot

#%%
# Creating values for the python functions
k_values = np.linspace(0, 0.04, 10000)
x = np.linspace(0, 0.04, 10000)
solution(k_values)

#%%
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(k_values, solution(k_values))
ax.plot(x, y(x))

plt.xlabel('$k_{t}$')
plt.ylabel('$k_{t+1}$')
plt.title('Key Transition Equation')

plt.show()

# 4.Implementation of the Bisection method at the steady state level of the model

#%%
# Creating the Key Transition Equation for the steady state
f = lambda k, alpha=0.4, n=0.01, g=0.06, beta=0.8: k - ((1-alpha)/((1+g1(n,g))*(1+1/beta)))** (1/(1-alpha))

#%%
# Implementing the Bisection method
def bisection(f,a,b,max_iter=500,tol=1e-6,full_info=False):
    """ bisection
        
    Args:
    
        f (function): function
        a (float): left bound
        b (float): right bound
        tol (float): tolerance on solution
        
    Returns:
    
        m (float): root
    
    """
    
    # test inputs
    if f(a)*f(b) >= 0:
        print("bisection method fails.")
        return None
    
    # step 1: initialize
    _a = a
    _b = b
    a = np.zeros(max_iter)
    b = np.zeros(max_iter)
    m = np.zeros(max_iter)
    fm = np.zeros(max_iter)
    a[0] = _a
    b[0] = _b
    
    # step 2-4: main
    i = 0
    while i < max_iter:
        
        # step 2: midpoint and associated value
        m[i] = (a[i]+b[i])/2
        fm[i] = f(m[i])
        
        # step 3: determine sub-interval
        if abs(fm[i]) < tol:
            break        
        elif f(a[i])*fm[i] < 0:
            a[i+1] = a[i]
            b[i+1] = m[i]
        elif f(b[i])*fm[i] < 0:
            a[i+1] = m[i]
            b[i+1] = b[i]
        else:
            print("bisection method fails.")
            return None
        
        i += 1
        
    if full_info:
        return m,i,a,b,fm
    else:
        return m[i],i

#%%
m,i = bisection(f,-10,10)
print(i,m,f(m))

#%%
def plot_bisection(f, a, b, xmin=-10, xmax=10, xn=100):
    
    # a. find root and return all information 
    m, max_iter, a, b, fm = bisection(f, a, b, full_info=True)
    
    # b. compute function on grid
    xvec = np.linspace(xmin, xmax, xn)
    fxvec = f(xvec)
    
    # c. figure
    def _figure(i):
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1,1,1)
        
        ax.plot(xvec,fxvec) # on grid
        ax.plot(m[i],fm[i],'o',color='black',label='current') # mid
        ax.plot([a[i],b[i]],[fm[i],fm[i]],'--',color='black',label='range') # range
        ax.axvline(a[i],ls='--',color='black')
        ax.axvline(b[i],ls='--',color='black')        
        
        ax.legend(loc='lower right')
        ax.grid(True)
        ax.set_ylim([fxvec[0],fxvec[-1]])
        
    widgets.interact(_figure,
        i=widgets.IntSlider(description="iterations", min=0, max=max_iter, step=1, value=0)
    );  

plot_bisection(f, -10, 10)