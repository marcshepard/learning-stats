"""basics.py - statistical testing if a distribution has a different mean"""

#pylint: disable-msg=invalid-name, line-too-long, import-error, too-many-arguments, pointless-string-statement
import numpy as np
from scipy import stats
import statsmodels.stats.power as smp
import matplotlib.pyplot as plt

"""
Consider an experiment to figure out if a distribution has a different mean than a null hypothesis.
For example, the null hypothesis might be that the click-through rate of a website will remain the
same after some new ad treatment. The alternative hypotheses is that the click-through rate will increase.
To test, we can run an experiment where we show the new ad to a sample of users and compare the click-through
rate of the sample to the click-through rate of the population. If the click-through rate of the sample is
significantly different than the click-through rate of the population, we can reject the null hypothesis
and conclude that the click-through rate has changed.

A frequentist approach is to calculate the p-value of the sample mean, which is the probability of getting
that sample mean if the null hypothesis is true. If the p-value is less than the alpha level (usually .05),
then frequentists conclude that the null hypothesis is likely false and the alternative hypothesis is likely true.
Although they will be wrong some percent of the time:
* If the null hypothesis is true, they will reject it 5% of the time (the alpha level). This is called a type 1 error
* If the alternative hypothesis is true, they will fail to reject the null hypothesis 20% of the time (beta level). This is called a type 2 error
The alpha level one can set at the start of the experiment.
The best level depends on how big an effect you need to detect (the bigger the effect the lower the beta), and how many
samples you are willing to get (the more samples the lower the beta).

The code below calculates the beta level of a test when sampling from a normal distribution for various effect sizes and sample sizes.
"""

def calculate_power (mu, mean, sd, n, sims=1000, alpha=.05, plot=False):
    """Calculate the power of a t-test
    mu = mean of the null hypothesis
    mean = mean of the distribution being tested
    sd = standard deviation of the distribution
    n = sample size
    alpha = alpha level
    """

    # Calculate the power of the test
    print (f"Calculating the power of a test comparing null hypothesis mu={mu} with {n} samples from a normal distribution of mean={mean} and standard deviation of={sd}")

    # Calcualte the expected power
    pw = smp.tt_solve_power(effect_size=(mean-mu)/sd, nobs=n, alpha=alpha, alternative="two-sided")
    print (f"Expected power = {pw:.2f}")

    # With a mean difference of 6, and SD of 15, and a sample size of 26, the test has 50% power
    # Let's see if we can get that with a simulation
    p = [] # set up empty variable to store all simulated p-values
    bars = 20
    for _ in range(sims):       #for each simulated experiment
        x = np.random.normal(mean, sd, n) #Simulate data with specified mean, standard deviation, and sample size
        z = stats.ttest_1samp(x, mu)   # 1 sample t-test against mu (set to value you want to test against)
        p.append (z[1]) #get the p-value and store it
    p = np.array(p)
    print(f"Simulation power after {sims} simulations = {sum(p < alpha)/sims:.2f}")

    if plot:
        # Plot the distribution of p-values
        plt.hist(p, bins=bars)
        plt.xlabel("P-values")
        plt.ylabel("number of p-values")
        plt.title(f"P-value Distribution with {round(sum(p < alpha)/sims*100, 1)}% Power")
        plt.axhline(y=sims/bars, color='r', linestyle='--')
        plt.show()

MU = 100    # null hypothesis mean
MEAN = 106  # Mean of the distribution being tested
SD = 15     # standard deviation of the distribution
N = 26     # sample size
SIMS = 1000 # number of simulated experiments
ALPHA = .05

calculate_power (MU, MEAN, SD, N, SIMS, ALPHA)

# Calculate the number of samples needed to reach a desired power level
POWER = .8
np.seterr(divide="ignore")
samples_needed = int(smp.tt_solve_power(effect_size=(MEAN-MU)/SD, power=POWER, alpha=ALPHA, alternative="two-sided") + .5)
print (f"To get a power level of {POWER}, one would need {samples_needed} samples")
