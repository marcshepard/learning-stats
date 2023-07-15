#pylint: disable=line-too-long, invalid-name, too-many-locals

""" Testing out bootstrap resampling 

Given an "unknown" population distribution, predict it's variance via bootstrap and the sample variance to see which is more accurate.
"""

import numpy as np
from scipy import stats

# Configurable parameters
POP_DIST = stats.binom(n=10, p=.1)  # The "unknown" population distribution. Change it to anything you want
SAMPLE_SIZE = 100                   # Number of samples to take from the population distribution - sample variance computed from his
BOOTSTRAP_SAMPLES = 1000            # Number of bootstrap samples to generate from the initial samples - to compute the bootstrap variance
N_EXPERIMENTS = 60                  # Conduct this many experiments and compare how far off the two predictions are (MAE)

# For reproducability
#np.random.seed(seed=2112)

def experiment(pop_dist, sample_size, n_bootstrap_samples, n_experiments):
    """ Run the experiment """
    print ("We'll start with an 'unknown' distribution, and give it's true mean and variance")
    print (f"Then we'll repeat the following experiment {n_experiments} times:")
    print (f"\tTake a sample of size {sample_size} and see what the bessels correction predicts as the population variance")
    print (f"\tUse bootstrapping, taking {n_bootstrap_samples} samples of size {sample_size} from the previous sample (with replacement) to see what it predicts as the population variance")
    print ()

    pop_mean = pop_dist.mean()
    pop_var = pop_dist.var()

    sample_var_loss = 0
    bootstrap_var_loss = 0

    for _ in range(n_experiments):
        # Print the true population mean and variance
        print (f"Population mean={pop_mean:.2f}, variance={pop_var:.2f}")

        # Take a sample and directly predict the population mean and variance from that usign bessel's correction
        sample = pop_dist.rvs (size=sample_size)
        sample_mean = np.mean(sample)
        sample_var = np.var(sample, ddof=1) # Sample variance with bessel's correction
        sample_var_loss += np.abs(sample_var - pop_var)
        print (f"Sample     mean={sample_mean:.2f}, variance={sample_var:.2f}")

        # Use bootstrapping to predict the population variance
        bootstrap_samples = np.random.choice(sample, size=(n_bootstrap_samples, sample_size), replace=True)
        bootstrap_sample_mean = np.mean(bootstrap_samples, axis=1)
        bootstrap_sample_variances = np.var(bootstrap_samples, ddof=1, axis=1)
        bootstrap_var = np.mean(bootstrap_sample_variances)
        bootstrap_var_loss += np.abs(bootstrap_var - pop_var)
        print (f"Bootstrap  mean={np.mean(bootstrap_sample_mean):.2f}, variance={bootstrap_var:.2f}")
        print ()

        # Sanity check my math, since I'm not getting the results I'm expected (which is that the bootstrap variance should be closer to the population variance)
        bootstrap_ci = stats.bootstrap((sample,), np.var, n_resamples=n_bootstrap_samples).confidence_interval
        assert bootstrap_ci[0] <= bootstrap_var <= bootstrap_ci[1], f"Bootstrap variance {bootstrap_var} not in confidence interval {bootstrap_ci}"

    print ("The average error for each technique:")
    print (f"Sample variance                       : {sample_var_loss/N_EXPERIMENTS:.2f}")
    print (f"Bootstrap variance                    : {bootstrap_var_loss/N_EXPERIMENTS:.2f}")

experiment(POP_DIST, SAMPLE_SIZE, BOOTSTRAP_SAMPLES, N_EXPERIMENTS)
