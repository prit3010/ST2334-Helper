from scipy import stats as st
import statistics as sta
import math

"""
This package contains functions specific for usage in statistics, and was designed
for students' use in ST2334, a statistics module by National University of Singapore.

The first section covers general functions for convenience.
"""

# Displays all input line by line
def disp(*answers):
    for ans in answers:
        print(ans)

# Takes in an array of tuples, of value and corresponding probability, and returns mean
def find_mu(pdf):
    res = 0
    for case in pdf:
        res += case[0] * case[1]
    return res

# Takes in an array of tuples, of value and corresponding probability, and returns variance
def find_var(pdf):
    ex2 = 0
    for case in pdf:
        ex2 += case[0] ** 2 * case[1]
    return ex2 - find_mu(pdf) ** 2

# Returns the cdf of x for the standard normal distribution
def z_cdf(x):
    return st.norm.cdf(x, 0, 1)

# Returns a lambda that transforms binom param to norm param
def binom_to_norm_transformer(n, p):
    res = lambda x: (x - n*p) / ((n * p * (1 - p)) ** 0.5)
    return res

# Returns a lambda that transforms norm param to t param
def norm_to_t_transformer(n, mu, var):
    res = lambda x: (x - mu) / (var ** 0.5 / n ** 0.5)
    return res

"""
The next section covers functions specifically for constructing confidence intervals.
"""

# Confidence Interval with Known population Variance
def ci_known(mean, conf_level, n, std):
    return [mean - (st.norm.ppf((1 - conf_level) / 2 + conf_level, 0, 1) * (std / math.sqrt(n))),
            mean + (st.norm.ppf((1 - conf_level) / 2 + conf_level, 0, 1) * (std / math.sqrt(n)))]

# Confidence Interval with Unknown population Variance
def ci_unknown(mean, conf_level, n, std):
    return [mean - (st.t.ppf((1 - conf_level) / 2 + conf_level, n - 1) * (std / math.sqrt(n))),
            mean + (st.t.ppf((1 - conf_level) / 2 + conf_level, n - 1) * (std / math.sqrt(n)))]


# Confidence Interval for Difference of Two Means(Known Variance)
# Mean, n and std must be array with 1, 2
def diff_ci_known(mean, conf_level, n, std):
    return [(mean[0] - mean[1]) - (st.norm.ppf((1 - conf_level) / 2 + conf_level, 0, 1) \
                    * (math.sqrt(std[0] ** 2 / n[0] + std[1] ** 2 / n[1]))),
            (mean[0] - mean[1]) + (st.norm.ppf((1 - conf_level) / 2 + conf_level, 0, 1) \
                    * (math.sqrt(std[0] ** 2 / n[0] + std[1] ** 2 / n[1])))]

# Confidence Interval for Difference of Two Means (Unknown Variance)
# + Population Variances Equal and
# for small Sample
def pooled_sample_var(n, var):
    return ((n[0] - 1) * var[0] + (n[1] - 1) * var[1]) \
            / (n[0] + n[1] - 2)

def diff_ci_unknown_small(mean, conf_level, n, std):
    var = pooled_sample_var(n, std)
    return [(mean[0] - mean[1]) - (st.t.ppf((1 - conf_level) / 2 + conf_level, n[0] + n[1] - 2) \
                    * (var * math.sqrt(1 / n[0] + 1 / n[1]))),
            (mean[0] - mean[1]) + (st.t.ppf((1 - conf_level) / 2 + conf_level, n[0] + n[1] - 2) \
                    * (var * math.sqrt(1 / n[0] + 1 / n[1])))]


# Confidence Interval for Difference of Two Means (Unknown Variance)
# + Population Variances Equal and
# for large Sample

def diff_ci_unknown_large(mean, conf_level, n, std):
    var = pooled_sample_var(n, std)
    return [(mean[0] - mean[1]) - (st.norm.ppf((1 - conf_level) / 2 + conf_level, 0, 1) \
                    * (var * math.sqrt(1 / n[0] + 1 / n[1]))),
            (mean[0] - mean[1]) + (st.norm.ppf((1 - conf_level) / 2 + conf_level, 0, 1) \
                    * (var * math.sqrt(1 / n[0] + 1 / n[1])))]

# Confidence Interval for Variances

# Case 1: Population Mean is known
def sum_squared_diff(entry, mean):
    result = 0
    for i in entry:
        result += (entry[i] - mean) ** 2
    return result

def var_ci_known(entry, mean, n, conf_level):
    return [sum_squared_diff(entry,mean) / st.chi2.ppf((1 - conf_level) / 2,n),
            sum_squared_diff(entry,mean) / st.chi2.ppf((1 - conf_level) / 2 + conf_level,n)]

# Case 2: Population Mean is unknown
def var_ci_unknown(sample_var, n, conf_level):
    return [(n - 1) * sample_var / st.chi2.ppf((1 - conf_level) / 2  + conf_level, n - 1),
           (n - 1) * sample_var / st.chi2.ppf((1 - conf_level) / 2 , n - 1)]

# C.I for ratio of two variance with Unknown Means
def ratio_var_ci_unknown(sample_var, n, conf_level):
    return [(sample_var[0] / sample_var[1]) * (1 / st.f.ppf((1 - conf_level) / 2 \
                    + conf_level, n[0] - 1, n[1] - 1)),
            (sample_var[0] / sample_var[1]) * (st.f.ppf((1 - conf_level) / 2 \
                    + conf_level, n[1] - 1, n[0] - 1))]

# p-value for two-tailed tests, given raw cdf for a distribution 
def p_value(p):
    return 2 * min(p, 1 - p)
