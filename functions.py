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
    print("")

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

# Takes in two arrays of paired data, and returns sample mean and variance
def paired_data(x, y):
    n = len(x)
    diff = [x[i] - y[i] for i in range(n)]
    d_bar = sta.mean(diff)
    var = sta.variance(diff)
    return [d_bar, var]

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
# Mean, n and std must be list with 1, 2
def diff_ci_known(mean, conf_level, n, var):
    return [(mean[0] - mean[1]) - (st.norm.ppf((1 - conf_level) / 2 + conf_level, 0, 1) \
                    * (math.sqrt(var[0] / n[0] + var[1] / n[1]))),
            (mean[0] - mean[1]) + (st.norm.ppf((1 - conf_level) / 2 + conf_level, 0, 1) \
                    * (math.sqrt(var[0] / n[0] + var[1] / n[1])))]

# Confidence Interval for Difference of Two Means (Unknown Variance)
# + Population Variances Equal
# Mean, n and var must be list with 1, 2
def diff_ci_unknown(mean, conf_level, n, var):
    prob = (1 - conf_level) / 2 + conf_level
    center = mean[0] - mean[1]
    factor = 0
    if n[0] >= 30 and n[1] >= 30:
        factor = st.t.ppf(prob, n[0] + n[1] - 2)
    else:
        factor = st.norm.ppf(prob, 0, 1)
    diff = factor * math.sqrt((var[0] / n[0]) + (var[1] / n[1]))
    return [center - diff, center + diff]

def pooled_sample_var(n, var):
    return ((n[0] - 1) * var[0] + (n[1] - 1) * var[1]) \
            / (n[0] + n[1] - 2)

def diff_ci_equal(mean, conf_level, n, var):
    var_p = pooled_sample_var(n, var)
    return diff_ci_unknown(mean, conf_level, n, [var_p, var_p])

# Confidence Interval for Paired Data
# sample mean and variance must be calculated first
def paired_ci_computed(d_bar, var, n, conf_level):
    t_value = st.t.ppf((1 - conf_level) / 2 + conf_level, n - 1)
    diff = t_value * (var / n) ** 0.5
    return [d_bar - diff, d_bar + diff]

def paired_ci_raw(x, y, conf_level):
    data = paired_data(x, y)
    return paired_ci_computed(data[0], data[1], len(x), conf_level)

# Confidence Interval for Variances

# Case 1: Population Mean is known
def sum_squared_diff(entry, mean):
    result = 0
    for i in entry:
        result += (entry[i] - mean) ** 2
    return result

def var_ci_known(entry, mean, n, conf_level):
    return [sum_squared_diff(entry, mean) / st.chi2.ppf((1 - conf_level) / 2,n),
            sum_squared_diff(entry, mean) / st.chi2.ppf((1 - conf_level) / 2 + conf_level, n)]

# Case 2: Population Mean is unknown
def var_ci_unknown(sample_var, n, conf_level):
    return [(n - 1) * sample_var / st.chi2.ppf((1 - conf_level) / 2  + conf_level, n - 1),
           (n - 1) * sample_var / st.chi2.ppf((1 - conf_level) / 2, n - 1)]

# C.I for ratio of two variance
def ratio_var_ci(sample_var, n, conf_level):
    return [(sample_var[0] / sample_var[1]) * (1 / st.f.ppf((1 - conf_level) / 2 \
                    + conf_level, n[0] - 1, n[1] - 1)),
            (sample_var[0] / sample_var[1]) * (st.f.ppf((1 - conf_level) / 2 \
                    + conf_level, n[1] - 1, n[0] - 1))]

"""
This section covers functions for hypothesis testing.
The implemented strategy here for such cases is to determine the value c such that
the cdf of a distribution up to c (or 1 - cdf) can be used to obtain the appropriate
p-value. As such, there are two main cases:

1. Single-tail: Use a transformer to find the correct value c, then use either
        cdf or 1 - cdf to get the p-value.
2. Double-tail: Use a transformer to find the correct value c, then obtain cdf,
        and apply p_value_two on the resultant probability.
"""

# p-value for two-tailed tests, given raw cdf for a distribution 
def p_value(p, tail):
    if tail == -1:
        return p
    elif tail == 1:
        return 1 - p
    elif tail == 0:
        return 2 * min(p, 1 - p)
    else:
        raise ValueError("Not a valid input of tails")

def comp_p_alpha(pv, alpha):
    if pv > alpha:
        return f"{pv} > {alpha}, H_0 is not rejected"
    else:
        return f"{pv} < {alpha}, H_0 is rejected"

# Returns a lambda that transforms binom param to norm param
def binom_norm_transformer(n, p):
    res = lambda x: (x - n*p) / ((n * p * (1 - p)) ** 0.5)
    return res

# Returns a lambda that transforms norm param to t param, can also be used for z param
def norm_t_transformer(n, mu, var):
    res = lambda x: (x - mu) / (var / n) ** 0.5
    return res

def diff_t_transformer(n, mu, var):
    res = lambda x_bar1, x_bar2: ((x_bar1 - x_bar2) - (mu[0] - mu[1])) \
            / ((var[0] / n[0]) + (var[1] / n[1])) ** 0.5
    return res

def norm_chi2_transformer(n, var):
    res = lambda x: (n - 1) * x / var
    return res

def norm_f_transformer(var1, var2):
    res = lambda s_sq1, s_sq2: (s_sq1 * var1) / (s_sq2 * var2)
    return res

def pv_calc(dist, n, mu, var, test_val, tails):
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

def pv_calc_f(n1, n2, var1, var2, s_sq1, s_sq2, tails):
    trans = norm_f_transformer(var1, var2)
    return p_value(st.f.cdf(trans(s_sq1, s_sq2), n1 - 1, n2 - 1), tails)

def mean_hypotest_known(x_bar, mu, var, n, alpha, tails):
    pv = pv_calc(st.norm, n, mu, var, x_bar, tails)
    return comp_p_alpha(pv, alpha)

def mean_hypotest_unknown(x_bar, mu, var, n, alpha, tails):
    pv = pv_calc(st.t, n, mu, var, x_bar, tails)
    return comp_p_alpha(pv, alpha)

def diff_hypotest_known(x_bar, mu, var, n, alpha, tails):
    trans = diff_t_transformer(n, mu, var)
    pv = pv_calc(st.norm, 1, 0, 1, trans(x_bar[0], x_bar[1]), tails)
    return comp_p_alpha(pv, alpha)

def diff_hypotest_unknown(x_bar, mu, var, n, alpha, tails):
    trans = diff_t_transformer(n, mu, var)
    pv = 0
    if n[0] >= 30 and n[1] >= 30:
        pv = p_value(z_cdf(trans(x_bar[0], x_bar[1])), tails)
    else:
        pv = p_value(st.t.cdf(trans(x_bar[0], x_bar[1]), n[0] + n[1] - 2), tails)
    return comp_p_alpha(pv, alpha)

def diff_hypotest_equal(x_bar, mu, var, n, alpha, tails):
    var_p = pooled_sample_var(n, var)
    return diff_hypotest_unknown(x_bar, mu, [var_p, var_p], n, alpha, tails)

def paired_hypotest_computed(d_bar, mu_d, var, n, alpha, tails):
    pv = pv_calc(st.t, n, mu_d, var, d_bar, tails)
    return comp_p_alpha(pv, alpha)

def paired_hypotest_raw(x, y, mu_d, alpha, tails):
    d_bar, var = paired_data(x, y)
    return paired_hypotest_computed(d_bar, mu_d, var, len(x), alpha, tails)

def var_hypotest(s_sq, var, n, alpha, tails):
    pv = pv_calc(st.chi2, n, 0, var, s_sq, tails)
    return comp_p_alpha(pv, alpha)

def var_ratio_hypotest(s_sq, var, n, alpha, tails):
    pv = pv_calc_f(n[0], n[1], var[0], var[1], s_sq[0], s_sq[1], tails)
    return comp_p_alpha(pv, alpha)
