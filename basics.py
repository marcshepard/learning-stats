import numpy as np
import scipy.stats as stats
import statsmodels.stats.power as smp
import matplotlib.pyplot as plt

sims = 100000 #number of simulated experiments
sims = 1000 #number of simulated experiments

# Parameters for creating a distribution
mean = 106  # Mean IQ score in the sample (will be compared with 100 in a one-sample t-test)
n = 26      # sample size
n = 52
sd = 15     # standard deviation

# null hypothesis and alpha
mu = 100    # null hypothesis mean
alpha = 0.05 # alpha level

# With a mean difference of 6, and SD of 15, and a sample size of 26, the test has 50% power
# Let's see if we can get that with a simulation
print (f"""Simulating {sims} experiments with a mean of {mean}, standard deviation of {sd}, and sample size of {n}""")
print ("To see how it compares to the null hypothesis of a mean of 100")
p = [] # set up empty variable to store all simulated p-values
bars = 20
for i in range(sims):       #for each simulated experiment
    x = np.random.normal(mean, sd, n) #Simulate data with specified mean, standard deviation, and sample size
    z = stats.ttest_1samp(x, mu)   # 1 sample t-test against mu (set to value you want to test against)
    p.append (z[1]) #get the p-value and store it
p = np.array(p)
print(f"Simulation Power = {sum(p < alpha)/sims}")

# Calcualte the expected power
pw = smp.tt_solve_power(effect_size=(mean-mu)/sd, nobs=n, alpha=alpha, alternative="two-sided")
print (f"Expected Power = {pw}")

# Plot the distribution of p-values
plt.hist(p, bins=bars)
plt.xlabel("P-values")
plt.ylabel("number of p-values")
plt.title(f"P-value Distribution with {round(sum(p < alpha)/sims*100, 1)}% Power")
plt.axhline(y=sims/bars, color='r', linestyle='--')
plt.show()
