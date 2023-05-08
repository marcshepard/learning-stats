"""Likelihood functions for use in Bayesian statistics."""

#pylint: disable-msg=invalid-name, line-too-long, import-error, too-many-arguments

from scipy.stats import binom, beta, ttest_ind
import matplotlib.pyplot as plt
import numpy as np

def plot_likelihoods (n : int, x : int, H0 : float = None, H1 : float = None) -> None:
    """Plot the likelihood of two hypotheses given the data
    n = number of trials
    x = number of successes
    H0 = hypothesis 0 of percentage of success
    H1 = hypothesis 1 of percentage of success
    """
    theta = np.linspace(0, 1, 100)
    like = binom.pmf(x, n, theta)
    plt.plot(theta, like)

    if H0 is not None:
        plt.plot(H0, binom.pmf(x, n, H0), "ro")
        plt.text(H0, binom.pmf(x, n, H0), "H0")
        plt.axhline(binom.pmf(x, n, H0), color='r', linestyle='--')

    if H1 is not None:
        plt.plot(H1, binom.pmf(x, n, H1), "bo")
        plt.text(H1, binom.pmf(x, n, H1), "H1")
        plt.axhline(binom.pmf(x, n, H1), color='b', linestyle='--')

    title = f"Likelihood curve for {x} successes in {n} binomial trials"
    if H0 is not None and H1 is not None:
        title += f"\nLikelihood of {H0:.2f} over {H1:.2f} = {binom.pmf(x, n, H0)/binom.pmf(x, n, H1):.2f}"
    plt.title (title)

    plt.show()

def compare_likelihoods (n : int, x : int, H0 : float, H1 : float) -> float:
    """Compare the likelihood of two hypotheses given the data
    n = number of trials
    x = number of successes
    H0 = hypothesis 0
    H1 = hypothesis 1

    Returns the likelihood of H0 over H1
    """
    h0 = binom.pmf(x, n, H0)
    h1 = binom.pmf(x, n, H1)
    return h0/h1

def likelyhood_experimentation():
    """Experiment with likelihood functions"""
    flips = 13
    heads = np.random.binomial(13, .5)

    print (f"Probability of {heads} heads in {flips} flips = {binom.pmf(heads, flips, .5)}")

    experiment_data = [] # tuples of the form (n, x, H0, H1)
    # Simulate flipping a coin 13 times and comparing the likelihood of what we saw vs a fair coin
    experiment_data.append((13, heads, .5, heads/13))

    # Simulate the relative likelyhood of H0=.5 vs H1=.4 for various samples
    experiment_data.append((10, 5, .5, .4))
    experiment_data.append((100, 50, .5, .4))
    experiment_data.append((1000, 500, .5, .4))

    for n, x, H0, H1 in experiment_data:
        print (f"Likelihood of {H0} over {H1} for {x}/{n} heads = {compare_likelihoods(n, x, H0, H1)}")
        #plot_likelihoods(n, x, H0, H1)

def binomialBayes (n : int, x : int, a_prior : float, b_prior : float, plot : bool = False, H0 = .5) -> None:
    """Calculate the posterior probability of a binomial distribution, and optionally plot the three graphs
    n = number of trials
    x = number of successes
    a_prior = alpha for the Beta distribution for the prior
    b_prior = beta for the Beta distribution for the prior
    plot = whether to plot the three graphs
    H0 = the hypothesis to examine"""
    a_observed = x + 1          # Alpha for the Beta distribution for the likelihood of observed data
    b_observed = n - x + 1      # Beta for the Beta distribution for the likelihood of observed data
    a_posterior = a_prior + a_observed - 1 # Posterior alpha
    b_posterior = b_prior + b_observed - 1 # Posterior beta

    if plot:
        theta = np.linspace(0, 1, 1000)  # Create theta range from 0 to 1
        prior = beta (a_prior, b_prior).pdf(theta)
        observed = beta (a_observed, b_observed).pdf(theta)
        posterior = beta (a_posterior, b_posterior).pdf(theta)
        bf = beta (a_posterior, b_posterior).pdf(H0)/beta(a_prior, b_prior).pdf(H0)

        plt.plot(theta, prior, "grey", label="Prior")
        plt.plot(theta, observed, "b--", label="Observed", )
        plt.plot(theta, posterior, "k", label="Posterior")
        plt.plot([H0, H0], [beta (a_prior, b_prior).pdf(H0), beta (a_posterior, b_posterior).pdf(H0)], "k--")
        plt.legend()

        plt.title(f"Probabilities after {x} successes in {n} trials\nBayes factor = {bf:.2f}")

        plt.xlabel("Theta (pct chance of heads)")
        plt.ylabel("Density")
        plt.show()

    return a_posterior/(a_posterior+b_posterior)

def plot_mean_and_credible_interval (a, b):
    """Plot the mean and credible interval for a beta distribution
    a = alpha parameter for the beta distribution
    b = beta parameter for the beta distribution"""
    theta = np.linspace(0, 1, 1000)
    plt.plot(theta, beta(a, b).pdf(theta))
    plt.axvline(a/(a+b), color="k", linestyle="--")
    plt.axvline(beta(a, b).ppf(.025), color="k", linestyle="--")
    plt.axvline(beta(a, b).ppf(.975), color="k", linestyle="--")
    plt.title(f"Mean: {a/(a+b):.2f}. 95% credible interval: {beta(a, b).ppf(.025):.2f} to {beta(a, b).ppf(.975):.2f}")
    plt.xlabel("Theta (pct chance of heads)")
    plt.ylabel("Density")
    plt.show()

def binomialBayesExperimentation():
    """Experiment with binomial Bayes"""

    # 10 heads out of 20 flips with no prior beleif
    binomialBayes (20, 10, 1, 1, True, H0=.5)

    # 90 out of 100 heads with strong prior belief
    binomialBayes (100, 90, 100, 100, True)

    # Credible interval
    plot_mean_and_credible_interval (11, 11)

# Plot single p value over time for experiment that keeps collecting more data
def plot_p_values_over_time(n, D, SD):
    """Plot p-values over time for an experiment that keeps collecting more data
    n = number of datapoints to collect after the first 10
    D = true effect size
    SD = true standard deviation"""

    start = 10 # don't generate r values for the first 10 datapoints

    p = np.zeros(n+start)
    x = np.zeros(n+start)
    y = np.zeros(n+start)

    for i in range(start, n+start):
        x[i] = np.random.normal(0, SD)
        y[i] = np.random.normal(D, SD)
        p[i] = ttest_ind(x[0:i], y[0:i], equal_var=True)[1]

    # Plot p-values starting at the 10th datapoint
    plt.plot(p[start+1:])
    plt.axhline(0.05, color="k", linestyle="--")
    plt.title(f"p-values over time\nLowest p-value was {min(p[start+1:]):.2f} at sample size {np.argmin(p[start+1:])+start+1}")
    plt.xlabel("Sample size")
    plt.ylabel("p-value")
    plt.show()

def optional_stopping_sim (n, looks, n_sims, alpha, d):
    """Simulate optional stopping
    n = number of datapoints to collect
    looks = number of times to look at the data
    n_sims = number of simulations
    alpha = alpha level
    d = true effect size (0 to simulate Type 1 errors)"""

    # Create matrix to store p-values
    p = np.zeros((n_sims, looks))

    # Loop through simulations
    for i in range(n_sims):
        # Generate data
        x = np.random.normal(0, 1, n)
        y = np.random.normal(d, 1, n)

        # Loop through looks
        for j in range(looks):
            # Perform t-test
            p[i, j] = ttest_ind(x[0:(j+1)*n//looks], y[0:(j+1)*n//looks], equal_var=True)[1]

    # Plot histogram of min p values across looks for each similation
    plt.subplot(1, 2, 1)
    p_min = p.min(axis=1)
    plt.hist(p_min, bins=100)
    plt.axhline(n_sims/100, color="k", linestyle="--")
    plt.xlabel("Min across looks per experiment")
    plt.ylabel("count")

    # Plot histogram of the average p value across looks for each similation
    plt.subplot(1, 2, 2)
    plt.hist(p, bins=100)
    plt.axhline(len(p)/100, color="k", linestyle="--")
    plt.xlabel("All looks")
    plt.ylabel("count")

    # Calculate the percent of p_min values less than alpha
    sig_p_min = int(sum(p_min<alpha)/len(p_min) * 100)
    # Calculate the percent of p values less than alpha
    sig_p = int(sum(p.flatten()<alpha)/len(p.flatten()) * 100)


    plt.suptitle(f"Histogram of p-values when stopping is optional\n% significant experiments={sig_p_min}. % significant p-values={sig_p}")
    plt.show()

def optional_stopping_experimentation():
    """Experiment with optional stopping to show this can inflate type 1 errors"""
    plot_p_values_over_time(2000, 0, 1)             # Even with no effect, you eventually get a significant p-value 
    optional_stopping_sim (100, 5, 10000, 0.05, 0)  # This shows the percent of false positives over a number of experiements using optional stopping 
    optional_stopping_sim (100, 5, 1000, 0.0158, 0) # Early stopping using https://en.wikipedia.org/wiki/Pocock_boundary adjusted alpha level


optional_stopping_sim (100, 5, 5000, 0.0158, 0)