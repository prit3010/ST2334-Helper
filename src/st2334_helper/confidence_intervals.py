from scipy import stats
from . import general

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
    factor = stats.norm.ppf(upper_bound(conf_level), 0, 1)
    res = (factor * (std / err)) ** 2
    return f"n >= {res}"

# Confidence Intervals for Mean

def mean_known(mean, conf_level, n, std):
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
    factor = stats.norm.ppf(upper_bound(conf_level), 0, 1)
    diff = factor * (std / (n ** 0.5))
    return [mean - diff, mean + diff]

def mean_unknown(mean, conf_level, n, sample_std):
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
        return mean_known(mean, conf_level, n, sample_std)
    else:
        factor = stats.t.ppf(upper_bound(conf_level), n - 1)
        diff = factor * (sample_std / (n ** 0.5))
        return [mean - diff, mean + diff]

def diff_known(mean, conf_level, n, var):
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
    factor = stats.norm.ppf(upper_bound(conf_level), 0, 1)
    diff = factor * ((var[0] / n[0]) + (var[1] / n[1])) ** 0.5
    center = mean[0] - mean[1]
    return [center - diff, center + diff]

def diff_unknown(mean, conf_level, n, sample_var):
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
        factor = stats.t.ppf(upper_bound(conf_level), n[0] + n[1] - 2)
    else:
        factor = stats.norm.ppf(upper_bound(conf_level), 0, 1)
    diff = factor * ((sample_var[0] / n[0]) + (sample_var[1] / n[1])) ** 0.5
    center = mean[0] - mean[1]
    return [center - diff, center + diff]

def diff_equal(mean, conf_level, n, sample_var):
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
    var_p = general.pooled_sample_var(n, sample_var)
    return diff_unknown(mean, conf_level, n, [var_p, var_p])

def paired(mean, conf_level, n, sample_var):
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
    t_value = stats.t.ppf(upper_bound(conf_level), n - 1)
    diff = t_value * (sample_var / n) ** 0.5
    return [mean - diff, mean + diff]

def paired_raw(x, y, conf_level):
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
    data = general.paired_data(x, y)
    return paired(data[0], conf_level, len(x), data[1])

# Confidence Interval for Variances

def var_known(entry, mu, conf_level):
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
    factor_low = stats.chi2.ppf(upper_bound(conf_level), n)
    factor_high = stats.chi2.ppf(lower_bound(conf_level), n)
    ssq = general.sum_squares(entry, mu)
    return [ssq / factor_low, ssq / factor_high]

def var_unknown(sample_var, n, conf_level):
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
    factor_low = stats.chi2.ppf(upper_bound(conf_level), n)
    factor_high = stats.chi2.ppf(lower_bound(conf_level), n)
    ssq = (n - 1) * sample_var
    return [ssq / factor_low, ssq / factor_high]

def ratio_var(sample_var, n, conf_level):
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
    factor_low = (stats.f.ppf(upper_bound(conf_level), n[0] - 1, n[1] - 1)) ** -1
    factor_high = stats.f.ppf(upper_bound(conf_level), n[1] - 1, n[0] - 1)
    ratio = sample_var[0] / sample_var[1]
    return [ratio * factor_low, ratio * factor_high]
