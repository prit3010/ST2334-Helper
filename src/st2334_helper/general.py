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

import statistics

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
    d_bar = statistics.mean(diff)
    var = statistics.variance(diff)
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
