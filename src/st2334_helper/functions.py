"""This package contains functions specific for usage in statistics, and
was designed for students' use in ST2334, a statistics module by the
National University of Singapore.

The package imports from three modules, SciPy, and the native statistics
and math modules.

Functions are segmented into three sections:

1   General Use
2   Confidence Intervals
3   Hypotheses Testing

Designed primarily for use in the latter two sections, it is recommended
that for computation of statistics outside of the mentioned fields, other
software such as a graphing calculator be used instead.
"""

from scipy import stats as st
import statistics as sta
import math

# Section 1: General Use
# ----------------------
def disp(*answers):
    """Prints variable number of input arguments on separate lines.
    Additionally, prints a blank line at the end. Mainly useful for
    displaying numerous answers in the same function.

    Parameters
    ----------
    answers : any, varargs
        The answers to be printed out.
    """
    for ans in answers:
        print(ans)
    print("")

def find_mu(pdf):
    """Computes the mean, or expectation, for a given probability
    distribution function.
    Used for discrete cases where the pdf is not even.

    Parameters
    ----------
    pdf : float(2)[]
        The pdf to be computed. To be given as an array of ordered pairs,
        of (value, probability), which can be given as tuples.

    Returns
    -------
    float
        The expectation of the pdf.
    """
    res = 0
    for case in pdf:
        res += case[0] * case[1]
    return res

def find_var(pdf):
    """Computes the variance for a given probability
    distribution function.
    Used for discrete cases where the pdf is not even.

    Parameters
    ----------
    pdf : Tuple[float, float][]
        The pdf to be computed. To be given as an array of ordered pairs,
        of (value, probability), which can be given as tuples.

    Returns
    -------
    float
        The variance of the pdf.
    """
    ex2 = 0
    for case in pdf:
        ex2 += case[0] ** 2 * case[1]
    return ex2 - find_mu(pdf) ** 2

def paired_data(x, y):
    """Computes the sample mean and variance for difference in paired
    data.

    Parameters
    ----------
    x : float[]
        The values of the first dataset, which is dependent on the other.

    y : float[]
        The values of the second dataset, which is dependent on the other.

    Returns
    -------
    Tuple[float, float]
        The sample mean and variance of the difference between x and y,
        given in the form of (mean, variance).
    """
    n = len(x)
    diff = [x[i] - y[i] for i in range(n)]
    d_bar = sta.mean(diff)
    var = sta.variance(diff)
    return (d_bar, var)

def pooled_sample_var(n, sample_var):
    """Computes the pooled sample variance for two samples with the same
    population variance.

    Parameters
    ----------
    n : int[]
        The size of the samples. Must be a list of size 2.

    sample_var : float[]
        The variance of the samples. Must be a list of size 2.

    Returns
    -------
    float
        The value of the pooled sample variance.
    """
    return ((n[0] - 1) * sample_var[0] + (n[1] - 1) * sample_var[1])\
            / (n[0] + n[1] - 2)

def sum_squares(entry, mu):
    """Computes the sum of squared difference to mean, given that the
    population mean is known.

    Parameters
    ----------
    entry : float[]
        The sample of data.

    mu : float
        The population mean.

    Returns
    -------
    float
        The value of the sum of squared difference to mean.
    """
    result = 0
    for x in entry:
        result += (x - mu) ** 2
    return result

# Section 2: Confidence Intervals
# -------------------------------

def upper_bound(conf_level):
    """Computes the probability value of the upper bound of the
    confidence interval to be built.

    Parameters
    ----------
    conf_level : float
        The value of the confidence level.

    Returns
    -------
    float
        The value of the upper bound of the interval.
    """
    return (1 - conf_level) / 2 + conf_level

def lower_bound(conf_level):
    """Computes the probability value of the lower bound of the
    confidence interval to be built.

    Parameters
    ----------
    conf_level : float
        The value of the confidence level.

    Returns
    -------
    float
        The value of the lower bound of the interval.
    """
    return (1 - conf_level) / 2

def error_min_size(conf_level, std, err):
    """Finds the minimum size required to ensure a low level of error involved
    with construction of confidence intervals.
    This can only be used for confidence intervals on population mean, and
    only if it approximates to a normal distribution.

    Parameters
    ----------
    conf_level : float
        The value of the confidence level.

    std : float
        The standard deviation of the population, or the sample.

    err : float
        The error that is desired, at most.

    Returns
    -------
    string
        The resultant inequality, which displays n, the sample size, and the
        minimum size required.
    """
    factor = st.norm.ppf(upper_bound(conf_level), 0, 1)
    res = (factor * (std / err)) ** 2
    return f"n >= {res}"

# Confidence Intervals for Mean

def ci_known(mean, conf_level, n, std):
    """Constructs a confidence interval for a population mean, given
    that the population variance is known.

    Parameters
    ----------
    mean : float
        The sample mean.

    conf_level : float
        The value of the confidence level.

    n : int
        The size of the sample.

    std : float
        The standard deviation of the population.

    Returns
    -------
    float[]
        The lower and upper bounds of the confidence interval respectively.
    """
    factor = st.norm.ppf(upper_bound(conf_level), 0, 1)
    diff = factor * (std / (n ** 0.5))
    return [mean - diff, mean + diff]

def ci_unknown(mean, conf_level, n, sample_std):
    """Constructs a confidence interval for a population mean, given
    that the population variance is unknown.

    Parameters
    ----------
    mean : float
        The sample mean.

    conf_level : float
        The value of the confidence level.

    n : int
        The size of the sample.

    sample_std : float
        The standard deviation of the sample.

    Returns
    -------
    float[]
        The lower and upper bounds of the confidence interval respectively.
    """
    if n >= 30:
        return ci_known(mean, conf_level, n, sample_std)
    else:
        factor = st.t.ppf(upper_bound(conf_level), n - 1)
        diff = factor * (sample_std / (n ** 0.5))
        return [mean - diff, mean + diff]

def diff_ci_known(mean, conf_level, n, var):
    """Constructs a confidence interval for the difference in mean for
    two populations, given that the population variances for both are
    known. Data for each statistic should be given in a consistent
    ordering.

    Parameters
    ----------
    mean : float[]
        The sample mean. Must be a list of size 2.

    conf_level : float
        The value of the confidence level.

    n : int[]
        The size of the samples. Must be a list of size 2.

    var : float
        The variance of the populations. Must be a list of size 2.

    Returns
    -------
    float[]
        The lower and upper bounds of the confidence interval respectively.
    """
    factor = st.norm.ppf(upper_bound(conf_level), n - 1)
    diff = factor * ((var[0] / n[0]) + (var[1] - n[1])) ** 0.5
    center = mean[0] - mean[1]
    return [center - diff, center + diff]

def diff_ci_unknown(mean, conf_level, n, sample_var):
    """Constructs a confidence interval for the difference in mean for
    two populations, given that the population variances for both are
    unknown. Data for each statistic should be given in a consistent
    ordering.

    Parameters
    ----------
    mean : float[]
        The sample mean. Must be a list of size 2.

    conf_level : float
        The value of the confidence level.

    n : int[]
        The size of the samples. Must be a list of size 2.

    sample_var : float
        The variance of the samples. Must be a list of size 2.

    Returns
    -------
    float[]
        The lower and upper bounds of the confidence interval respectively.
    """
    factor = 0
    if n[0] >= 30 and n[1] >= 30:
        factor = st.t.ppf(upper_bound(conf_level), n[0] + n[1] - 2)
    else:
        factor = st.norm.ppf(upper_bound(conf_level), 0, 1)
    diff = factor * math.sqrt((sample_var[0] / n[0]) + (sample_var[1] / n[1]))
    center = mean[0] - mean[1]
    return [center - diff, center + diff]

def diff_ci_equal(mean, conf_level, n, sample_var):
    """Constructs a confidence interval for the difference in mean for
    two populations, given that the population variances for both are
    unknown but equal. Data for each statistic should be given in a
    consistent ordering.

    Parameters
    ----------
    mean : float[]
        The sample mean. Must be a list of size 2.

    conf_level : float
        The value of the confidence level.

    n : int[]
        The size of the samples. Must be a list of size 2.

    sample_var : float
        The variance of the samples. Must be a list of size 2.

    Returns
    -------
    float[]
        The lower and upper bounds of the confidence interval respectively.
    """
    var_p = pooled_sample_var(n, sample_var)
    return diff_ci_unknown(mean, conf_level, n, [var_p, var_p])

def paired_ci(mean, conf_level, n, sample_var):
    """Constructs a confidence interval for the difference in mean for
    two populations, given a set of paired data that is dependent. Data
    for each statistic should be given in a consistent ordering.

    Parameters
    ----------
    mean : float
        The sample mean.

    conf_level : float
        The value of the confidence level.

    n : int
        The size of the samples.

    sample_var : float
        The variance of the samples.

    Returns
    -------
    float[]
        The lower and upper bounds of the confidence interval respectively.
    """
    t_value = st.t.ppf(upper_bound(conf_level), n - 1)
    diff = t_value * (sample_var / n) ** 0.5
    return [mean - diff, mean + diff]

def paired_ci_raw(x, y, conf_level):
    """Constructs a confidence interval for the difference in mean for
    two populations, given a set of paired data that is dependent. This
    raw version is for cases where only the raw data is available. Data
    should be provided as lists with matching order and indexing.

    Parameters
    ----------
    x : float[]
        The values of the first data set. Must match the second set.

    y : float[]
        The values of the second data set. Must match the first set.

    conf_level : float
        The value of the confidence level.

    Returns
    -------
    float[]
        The lower and upper bounds of the confidence interval respectively.
    """
    data = paired_data(x, y)
    return paired_ci(data[0], data[1], len(x), conf_level)

# Confidence Interval for Variances

def var_ci_known(entry, mu, conf_level):
    """Constructs a confidence interval for the variance of a population,
    given that the population mean is known. The full data set has to be
    provided to accurately compute.

    Parameters
    ----------
    entry : float[]
        The sample of data.

    mu : float
        The population mean.

    conf_level : float
        The value of the confidence level.

    Returns
    -------
    float[]
        The lower and upper bounds of the confidence interval respectively.
    """
    n = len(entry)
    factor_low = st.chi2.ppf(upper_bound(conf_level), n)
    factor_high = st.chi2.ppf(lower_bound(conf_level), n)
    ssq = sum_squares(entry, mu)
    return [ssq / factor_low, ssq / factor_high]

def var_ci_unknown(sample_var, n, conf_level):
    """Constructs a confidence interval for the variance of a population,
    given that the population mean is unknown.

    Parameters
    ----------
    sample_var : float
        The variance of the sample.

    n : int
        The size of the sample.

    conf_level : float
        The value of the confidence level.

    Returns
    -------
    float[]
        The lower and upper bounds of the confidence interval respectively.
    """
    factor_low = st.chi2.ppf(upper_bound(conf_level), n)
    factor_high = st.chi2.ppf(lower_bound(conf_level), n)
    ssq = (n - 1) * sample_var
    return [ssq / factor_low, ssq / factor_high]

def ratio_var_ci(sample_var, n, conf_level):
    """Constructs a confidence interval for the ratio of variances of
    two populations.

    Parameters
    ----------
    sample_var : float[]
        The variance of the samples. Must be a list of size 2.

    n : int
        The size of the sample.

    conf_level : float
        The value of the confidence level.

    Returns
    -------
    float[]
        The lower and upper bounds of the confidence interval respectively.
    """
    factor_low = (st.f.ppf(upper_bound(conf_level), n[0] - 1, n[1] - 1)) ** -1
    factor_high = st.f.ppf(upper_bound(conf_level), n[1] - 1, n[0] - 1)
    ratio = sample_var[0] / sample_var[1]
    return [ratio * factor_low, ratio * factor_high]

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
    if dist == st.norm:
        return p_value(dist.cdf(test_val, mu, (var / n) ** 0.5), tails)
    elif dist == st.t:
        trans = norm_t_transformer(n, mu, var)
        return p_value(dist.cdf(trans(test_val), n - 1), tails)
    elif dist == st.chi2:
        trans = norm_chi2_transformer(n, var)
        return p_value(dist.cdf(trans(test_val), n - 1), tails)
    elif dist == st.f:
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
    return p_value(st.f.cdf(trans(sample_var1, sample_var2), n1 - 1, n2 - 1), tails)

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

def mean_hypotest_known(x_bar, mu, var, n, alpha, tails):
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
    pv = pv_calc(st.norm, n, mu, var, x_bar, tails)
    return comp_p_alpha(pv, alpha)

def mean_hypotest_unknown(x_bar, mu, sample_var, n, alpha, tails):
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
        dist = st.norm
    else:
        dist = st.t
    pv = pv_calc(dist, n, mu, sample_var, x_bar, tails)
    return comp_p_alpha(pv, alpha)

def diff_hypotest_known(x_bar, mu, var, n, alpha, tails):
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
    pv = pv_calc(st.norm, 1, 0, 1, trans(x_bar[0], x_bar[1]), tails)
    return comp_p_alpha(pv, alpha)

def diff_hypotest_unknown(x_bar, mu, sample_var, n, alpha, tails):
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
        pv = p_value(st.norm.cdf(trans(x_bar[0], x_bar[1])), tails)
    else:
        pv = p_value(st.t.cdf(trans(x_bar[0], x_bar[1]), n[0] + n[1] - 2), tails)
    return comp_p_alpha(pv, alpha)

def diff_hypotest_equal(x_bar, mu, sample_var, n, alpha, tails):
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
    var_p = pooled_sample_var(n, sample_var)
    return diff_hypotest_unknown(x_bar, mu, [var_p, var_p], n, alpha, tails)

def paired_hypotest(d_bar, mu_d, var, n, alpha, tails):
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
    pv = pv_calc(st.t, n, mu_d, var, d_bar, tails)
    return comp_p_alpha(pv, alpha)

def paired_hypotest_raw(x, y, mu_d, alpha, tails):
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
    d_bar, var = paired_data(x, y)
    return paired_hypotest(d_bar, mu_d, var, len(x), alpha, tails)

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
    pv = pv_calc(st.chi2, n, 0, var, sample_var, tails)
    return comp_p_alpha(pv, alpha)

def var_ratio_hypotest(sample_var, var, n, alpha, tails):
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
