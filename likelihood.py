"""Likelihood functions for use in Bayesian statistics."""

#pylint: disable-msg=invalid-name, line-too-long, import-error, too-many-arguments

from scipy.stats import binom, beta
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

#likelyhood_experimentation()

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

# 10 heads out of 20 flips with no prior beleif
#binomialBayes (20, 10, 1, 1, True)

# 90 out of 100 heads with strong prior belief
#binomialBayes (100, 90, 100, 100, True)

plot_mean_and_credible_interval (11, 11)
