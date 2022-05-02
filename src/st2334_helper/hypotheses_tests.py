from scipy import stats
from . import general

# Section 3: Hypotheses Testing
# -----------------------------

def p_value(p, tail):
    """Computes the appropriate p-value given the probability of an
    event, and whether it is a two-tailed or one-tailed test.

    Parameters
    ----------
    p : float
        The probability of the event.

    tail : int
        Indicates which case of hypothesis test it is. If the test is
        two-tailed, then the variable \'tail\' should be 0. If the
        test is one-tailed and the less than the value specified in the
        null hypothesis, the variable \'tail\' should be -1. Otherwise,
        it should be 1.

    Returns
    -------
    float
        The p-value of sample for the test.
    """
    if tail == -1:
        return p
    elif tail == 1:
        return 1 - p
    elif tail == 0:
        return 2 * min(p, 1 - p)
    else:
        raise ValueError("Not a valid input of tails")

def pv_calc(dist, n, mu, var, test_val, tails):
    """Computes the p-value for a test. Can be used with the normal
    distribution, t-distribution, chi2-distribution, but not the f-
    distribution.

    Parameters
    ----------
    dist : class
        The specific distribution to be used for the test.

    n : int
        The size of the sample.

    mu : float
        The population mean, hypothesized or known.

    var : float
        The population variance, hypothesized or known.

    test_val : float
        The value that is obtained from data, and to be tested.

    tails : int
        Indicates which tailed hypothesis test.

    Returns
    -------
    float
        The appropriate p-value for the test.
    """
    if dist == stats.norm:
        return p_value(dist.cdf(test_val, mu, (var / n) ** 0.5), tails)
    elif dist == stats.t:
        trans = norm_t_transformer(n, mu, var)
        return p_value(dist.cdf(trans(test_val), n - 1), tails)
    elif dist == stats.chi2:
        trans = norm_chi2_transformer(n, var)
        return p_value(dist.cdf(trans(test_val), n - 1), tails)
    elif dist == stats.f:
        raise ValueError("Use the function \'pv_calc_f\' instead")
    else:
        raise ValueError("Not a valid distribution")

def pv_calc_f(n1, n2, var1, var2, sample_var1, sample_var2, tails):
    """Computes the p-value for a test. Can only be used for the f-
    distribution.

    Parameters
    ----------
    n1 : int
        The size of the first sample.

    n2 : int
        The size of the second sample.

    var1 : float
        The hypothesized variance of the first population.

    var2 : float
        The hypothesized variance of the second population.

    sample_var1 : float
        The sum of squares for the first sample.

    sample_var2 : float
        The sum of squares for the second sample.

    tails : int
        Indicates which tailed hypothesis test.

    Returns
    -------
    float
        The appropriate p-value for the test.
    """
    trans = norm_f_transformer(var1, var2)
    return p_value(stats.f.cdf(trans(sample_var1, sample_var2), n1 - 1, n2 - 1), tails)

def comp_p_alpha(pv, alpha):
    """Compares a given p-value with a given level of significance, and
    returns the test conclusion.

    Parameters
    ----------
    pv : float
        The p-value of the test.

    alpha : float
        The level of significance of the test.

    Returns
    -------
    string
        The conclusion of the Hypothesis Test.
    """
    if pv > alpha:
        return f"{pv} > {alpha}, H_0 is not rejected"
    else:
        return f"{pv} < {alpha}, H_0 is rejected"

def binom_norm_transformer(n, p):
    """Provides a unary function to transform a binomial statistic to
    the standard normal distribution.

    Parameters
    ----------
    n : int
        The size parameter, typically n, of the binomial distribution.

    p : float
        The probability of a success of the binomial distribution.

    Returns
    -------
    lambda
        A unary function.
    """
    res = lambda x: (x - n*p) / ((n * p * (1 - p)) ** 0.5)
    return res

def norm_t_transformer(n, mu, sample_var):
    """Provides a unary function to transform a normal statistic to
    the t-distribution, with a specific degree of freedom.

    Parameters
    ----------
    n : int
        The size of the sample.

    mu : float
        The hypothesized mean of the population.

    sample_var : float
        The variance of the sample.

    Returns
    -------
    lambda
        A unary function.
    """
    res = lambda x: (x - mu) / (sample_var / n) ** 0.5
    return res

def diff_t_transformer(n, mu, var):
    """Provides a binary function to transform two normal statistics to
    the t-distribution, with a specific degree of freedom. Can be used
    regardless of whether population variance is known or unknown.

    Parameters
    ----------
    n : int[]
        The size of the samples. Must be given as a list of size 2.

    mu : float[]
        The hypothesized mean of the populations. Must be a list of size 2.

    var : float[]
        The variance of the sample or population. Must be a list of size 2.

    Returns
    -------
    lambda
        A binary function.
    """
    res = lambda x_bar1, x_bar2: ((x_bar1 - x_bar2) - (mu[0] - mu[1])) \
            / ((var[0] / n[0]) + (var[1] / n[1])) ** 0.5
    return res

def norm_chi2_transformer(n, var):
    """Provides a unary function to transform a normal statistic to
    the chi2-distribution, with a specific degree of freedom.

    Parameters
    ----------
    n : int
        The size of the sample.

    var : float
        The hypothesized variance of the population.

    Returns
    -------
    lambda
        A unary function.
    """
    res = lambda x: (n - 1) * x / var
    return res

def norm_f_transformer(var1, var2):
    """Provides a binary function to transform two sum of squares to
    the f-distribution, with specific degrees of freedom.

    Parameters
    ----------
    var1 : float
        The hypothesized variance of the first population.

    var2 : float
        The hypothesized variance of the second population.

    Returns
    -------
    lambda
        A binary function.
    """
    res = lambda s_sq1, s_sq2: (s_sq1 * var1) / (s_sq2 * var2)
    return res

def mean_known(x_bar, mu, var, n, alpha, tails):
    """Conducts a hypothesis test on the mean of a sample, given that
    the population variance is known.

    Parameters
    ----------
    x_bar : float
        The mean of the sample.

    mu : float
        The hypothesized mean of the population.

    var : float
        The variance of the population.

    n : int
        The size of the sample.

    alpha : float
        The level of significance of the test.

    tails : int
        Indicates which tailed hypothesis test it is.

    Returns
    -------
    string
        The conclusion of the test.
    """
    pv = pv_calc(stats.norm, n, mu, var, x_bar, tails)
    return comp_p_alpha(pv, alpha)

def mean_unknown(x_bar, mu, sample_var, n, alpha, tails):
    """Conducts a hypothesis test on the mean of a sample, given that
    the population variance is unknown.

    Parameters
    ----------
    x_bar : float
        The mean of the sample.

    mu : float
        The hypothesized mean of the population.

    sample_var : float
        The variance of the sample.

    n : int
        The size of the sample.

    alpha : float
        The level of significance of the test.

    tails : int
        Indicates which tailed hypothesis test it is.

    Returns
    -------
    string
        The conclusion of the test.
    """
    if n >= 30:
        dist = stats.norm
    else:
        dist = stats.t
    pv = pv_calc(dist, n, mu, sample_var, x_bar, tails)
    return comp_p_alpha(pv, alpha)

def diff_known(x_bar, mu, var, n, alpha, tails):
    """Conducts a hypothesis test on the difference of means of two
    samples, given that the population variance are known.

    Parameters
    ----------
    x_bar : float[]
        The mean of the samples. Must be a list of size 2.

    mu : float[]
        The hypothesized mean of the populations. Must be a list of
        size 2. If no specific mean are known, then only the relative
        difference in means is required to be accurate.

    var : float[]
        The variance of the populations. Must be a list of size 2.

    n : int[]
        The size of the samples. Must be a list of size 2.

    alpha : float
        The level of significance of the test.

    tails : int
        Indicates which tailed hypothesis test it is.

    Returns
    -------
    string
        The conclusion of the test.
    """
    trans = diff_t_transformer(n, mu, var)
    pv = pv_calc(stats.norm, 1, 0, 1, trans(x_bar[0], x_bar[1]), tails)
    return comp_p_alpha(pv, alpha)

def diff_unknown(x_bar, mu, sample_var, n, alpha, tails):
    """Conducts a hypothesis test on the difference of means of two
    samples, given that the population variance are unknown.

    Parameters
    ----------
    x_bar : float[]
        The mean of the samples. Must be a list of size 2.

    mu : float[]
        The hypothesized mean of the populations. Must be a list of
        size 2. If no specific mean are known, then only the relative
        difference in means is required to be accurate.

    sample_var : float[]
        The variance of the samples. Must be a list of size 2.

    n : int[]
        The size of the samples. Must be a list of size 2.

    alpha : float
        The level of significance of the test.

    tails : int
        Indicates which tailed hypothesis test it is.

    Returns
    -------
    string
        The conclusion of the test.
    """
    trans = diff_t_transformer(n, mu, sample_var)
    pv = 0
    if n[0] >= 30 and n[1] >= 30:
        pv = p_value(stats.norm.cdf(trans(x_bar[0], x_bar[1])), tails)
    else:
        pv = p_value(stats.t.cdf(trans(x_bar[0], x_bar[1]), n[0] + n[1] - 2), tails)
    return comp_p_alpha(pv, alpha)

def diff_equal(x_bar, mu, sample_var, n, alpha, tails):
    """Conducts a hypothesis test on the difference of means of two
    samples, given that the population variance are unknown but equal.

    Parameters
    ----------
    x_bar : float[]
        The mean of the samples. Must be a list of size 2.

    mu : float[]
        The hypothesized mean of the populations. Must be a list of
        size 2. If no specific mean are known, then only the relative
        difference in means is required to be accurate.

    sample_var : float[]
        The variance of the samples. Must be a list of size 2.

    n : int[]
        The size of the samples. Must be a list of size 2.

    alpha : float
        The level of significance of the test.

    tails : int
        Indicates which tailed hypothesis test it is.

    Returns
    -------
    string
        The conclusion of the test.
    """
    var_p = general.pooled_sample_var(n, sample_var)
    return diff_unknown(x_bar, mu, [var_p, var_p], n, alpha, tails)

def paired(d_bar, mu_d, var, n, alpha, tails):
    """Conducts a hypothesis test on the difference of means of two
    populations, given a set of paired data.

    Parameters
    ----------
    d_bar : float
        The mean of the difference of paired data.

    mu_d : float
        The hypothesized mean of difference in the populations.

    var : float
        The variance of the samples.

    n : int
        The size of the samples.

    alpha : float
        The level of significance of the test.

    tails : int
        Indicates which tailed hypothesis test it is.

    Returns
    -------
    string
        The conclusion of the test.
    """
    pv = pv_calc(stats.t, n, mu_d, var, d_bar, tails)
    return comp_p_alpha(pv, alpha)

def paired_raw(x, y, mu_d, alpha, tails):
    """Conducts a hypothesis test on the difference of means of two
    populations, given a set of paired data. This raw version is provided
    for cases where only the raw data is available.

    Parameters
    ----------
    x : float[]
        The first set of data. Must match with the second set.

    y : float[]
        The second set of data. Must match with the first set.

    mu_d : float
        The hypothesized mean of difference in the populations.

    alpha : float
        The level of significance of the test.

    tails : int
        Indicates which tailed hypothesis test it is.

    Returns
    -------
    string
        The conclusion of the test.
    """
    d_bar, var = general.paired_data(x, y)
    return paired(d_bar, mu_d, var, len(x), alpha, tails)

def var_hypotest(sample_var, var, n, alpha, tails):
    """Conducts a hypothesis test on the variance of a sample.

    Parameters
    ----------
    sample_var : float
        The variance of the sample.

    var : float
        The hypothesized variance of the population.

    n : int
        The size of the sample.

    alpha : float
        The level of significance of the test.

    tails : int
        Indicates which tailed hypothesis test it is.

    Returns
    -------
    string
        The conclusion of the test.
    """
    pv = pv_calc(stats.chi2, n, 0, var, sample_var, tails)
    return comp_p_alpha(pv, alpha)

def var_ratio(sample_var, var, n, alpha, tails):
    """Conducts a hypothesis test on the ratio of variance of two
    populations.

    Parameters
    ----------
    sample_var : float[]
        The variance of the samples. Must be a list of size 2.

    var : float[]
        The hypothesized variance of the population. Must be a list of
        size 2. If the specific variance are not known, then only the
        relative variance provided has to be accurate.

    n : int[]
        The size of the samples. Must be a list of size 2.

    alpha : float
        The level of significance of the test.

    tails : int
        Indicates which tailed hypothesis test it is.

    Returns
    -------
    string
        The conclusion of the test.
    """
    pv = pv_calc_f(n[0], n[1], var[0], var[1], sample_var[0], sample_var[1], tails)
    return comp_p_alpha(pv, alpha)
