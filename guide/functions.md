# Functions of ST2334-Helper

There are three main functionalities provided.

1. General Usage: basic and convenient functions, but very limited
2. Confidence Intervals: construct confidence intervals for samples
3. Hypotheses Testing: conduct hypothesis test given data

The detailed specifications and instructions are provided below.

## Navigation

1. [General Usage](#general-usage)

    - [Basic Distributions](#basic-distributions)
        - [Binomial Distribution](#binomial-distribution)
        - [Negative Binomial Distribution](#negative-binomial-distribution)
        - [Poisson Distribution](#poisson-distribution)
        - [Exponential Distribution](#exponential-distribution)
        - [Normal Distribution](#normal-distribution)
        - [t-Distribution](#t-distribution)
        - [Chi-squared Distribution](#chi-squared-distribution)
        - [f-Distribution](#f-distribution)
    - [Displaying answers](#displaying-answers)
    - [Discrete pdf](#discrete-pdf)
    - [Paired Data](#paired-data)
    - [Pooled Sample Variance](#pooled-sample-variance)
    - [Sum of Squares](#sum-of-squares)

2. [Confidence Intervals](#confidence-intervals)

    - [Interval Bounds](#interval-bounds)
    - [Minimum Size for a Desired Error](#minimum-size-for-a-desired-error)
    - [CI for Mean](#ci-for-mean)
        - [CI: Mean with Known Variance](#ci-mean-with-known-variance)
        - [CI: Mean with Unknown Variance](#ci-mean-with-unknown-variance)
        - [CI: Difference in Mean with Known Variance](#ci-difference-in-mean-with-known-variance)
        - [CI: Difference in Mean with Unknown Variance](#ci-difference-in-mean-with-unknown-variance)
        - [CI: Difference in Mean with Unknown but Equal Variance](#ci-difference-in-mean-with-unknown-but-equal-variance)
        - [CI: Difference in Mean of Paired Data](#ci-difference-in-mean-of-paired-data)
    - [CI for Variance](#ci-for-variance)
        - [CI: Variance with Known Mean](#ci-variance-with-known-mean)
        - [CI: Variance with Unknown Mean](#ci-variance-with-unknown-mean)
        - [CI: Ratio of Variance](#ci-ratio-of-variance)

3. [Hypotheses Testing](#hypotheses-testing)

    - [p-value](#p-value)
    - [Concluding the Test](#concluding-the-test)
    - [Transformers](#transformers)
        - [Binomial to Normal Transformer](#binomial-to-normal-transformer)
        - [Normal to t Transformer](#normal-to-t-transformer)
        - [Difference in Mean to t Transformer](#difference-in-mean-to-t-transformer)
        - [Normal to Chi-squared Transformer](#normal-to-chi-squared-transformer)
        - [Normal to f Transformer](#normal-to-f-transformer)
    - [Hypotheses Test for Mean](#hypotheses-test-for-mean)
        - [HT: Mean with Known Variance](#ht-mean-with-known-variance)
        - [HT: Mean with Unknown Variance](#ht-mean-with-unknown-variance)
        - [HT: Difference in Mean with Known Variance](#ht-difference-in-mean-with-known-variance)
        - [HT: Difference in Mean with Unknown Variance](#ht-difference-in-mean-with-unknown-variance)
        - [HT: Difference in Mean with Unknown but Equal Variance](#ht-difference-in-mean-with-unknown-but-equal-variance)
        - [HT: Difference in Mean of Paired Data](#ht-difference-in-mean-of-paired-data)
    - [Hypotheses Test for Variance](#hypotheses-test-for-variance)
        - [HT: Variance](#ht-variance)
        - [HT: Ratio of Variance](#ht-ratio-of-variance)

# General Usage

# Basic Distributions

These are functions already provided by **SciPy**, for computations using common distributions. The usage of these functions is not modified, but merely placed here for the user's reference.

## Binomial Distribution

Below are the functions for computations with the **Binomial Distribution**.

```python
from scipy import stats as st
# X ~ B(10,0.4), where X = number of successes, with number of trials = 10 and prob of a success = 0.4

# To find Pr(X <= 5),
st.binom.cdf(5,10,0.4) # gives 0.833761

# To find Pr(X = 5),
st.binom.pmf(5,10,0.4) # gives 0.200658

# To find Pr(X > 5),
1 - st.binom.cdf(5,10,0.4) # gives 0.166239

# To find x such that Pr(X <= x) >= 0.05,
st.binom.ppf(0.05,10,0.4) # gives 2
```

## Negative Binomial Distribution

Below are the functions for computations with the **Negative Binomial Distribution**.

```python
from scipy import stats as st
# X ~ NB(4,0.55), where X = number of trials, with number of successes = 4 and prob of a success = 0.55

# To find Pr(X <= 6),
st.nbinom.cdf(2,4,0.55) # gives 0.441518 , where 2 = number of failures

# To find Pr(X = 6),
st.nbinom.pmf(2,4,0.55) # gives 0.1853

# To find Pr(X > 6),
1 - st.nbinom.cdf(2,4,0.55) # gives 0.558482

# To find x such that Pr(X <= x) >= 0.25,
st.binom.ppf(0.25,4,0.55) # gives 1  which is the number of failures. Hence, x = 5
```

## Poisson Distribution

Below are the functions for computations with the **Poisson Distribution**.

```python
from scipy import stats as st
# X ~ P(8), where E(X) = lambda = 8

# To find Pr(X <= 6),
st.poisson.cdf(6,8) # gives 0.313374

# To find Pr(X = 6),
st.poisson.pmf(6,8) # gives 0.122138

# To find Pr(X > 6),
1 - st.poisson.cdf(6,8) # gives 0.686626

# To find x such that Pr(X <= x) >= 0.25,
st.poisson.ppf(0.25,8) # gives 6
```

## Exponential Distribution

Below are the functions for computations with the **Exponential Distribution**.

```python
from scipy import stats as st
# X ~ Exp(1/5), where E(X) = 5

# To find Pr(X <= 8),
st.expon.cdf(8,0,5) # gives 0.798103 with the second argument being the lower limit of the x range and 3rd argument = E(X)

# To find pdf f(8),
st.expon.pdf(8,0,5) # gives 0.0403793

# To find Pr(X > 8),
1 - st.expon.cdf(8,0,5) # gives 0.201897

# To find x such that Pr(X <= x) = 0.05,
st.expon.ppf(0.05,0,5) # gives 0.256466
```

## Normal Distribution

Below are the functions for computations with the **Normal Distribution**.

```python
from scipy import stats as st
# X ~ N(50, 10^2), where mu=E(X)=50 and sigma^2=V(X)=10^2

# To find Pr(X <= 45),
st.norm.cdf(45,50,10) # gives 0.308538

# To find pdf f(45),
st.norm.pdf(45,50,10) # gives 0.0352065

# To find Pr(X > 45),
1 - st.norm.cdf(45,50,10) # gives 0.691462

# To find x such that Pr(X <= x) = 0.05,
st.norm.ppf(0.05,50,10) # gives 33.5515

# To find z such that Pr(Z >= z) = 0.05 with Z ~ N(0,1),
st.norm.ppf(0.95,0,1) # gives 1.64485
```

## t-Distribution

Below are the functions for computations with the **t-Distribution**.

```python
from scipy import stats as st
# X ~ t(10), where degrees of freedom = 10

# To find Pr(X <= 1.5),
st.t.cdf(1.5,10) # gives 0.917746

# To find pdf f(1.5),
st.t.pdf(1.5,10) # gives 0.127445

# To find Pr(X > 1.5),
1 - st.t.cdf(1.5,10) # gives 0.0822537

# To find x such that Pr(X <= x) = 0.05,
st.t.ppf(0.05,10) # gives -1.81246

# To find x such that Pr(X >= x) = 0.05,
st.t.ppf(0.95,10) # gives 1.81246
```

## Chi-squared Distribution

Below are the functions for computations with the **Chi-squared Distribution**.

```python
from scipy import stats as st
# X ~ Chisq(10), where degrees of freedom = 10

# To find Pr(X <= 12),
st.chi2.cdf(12,10) # gives 0.714943

# To find pdf f(12),
st.chi2.pdf(12,10) # gives 0.0669263

# To find Pr(X > 12),
1 - st.chi2.cdf(12,10) # gives 0.285057

# To find x such that Pr(X <= x) = 0.05,
st.chi2.ppf(0.05,10) # gives 3.9403

# To find x such that Pr(X >= x) = 0.05,
st.chi2.ppf(0.95,10) # gives 18.307
```

## f-Distribution

Below are the functions for computations with the **f-Distribution**.

```python
from scipy import stats as st
# X ~ F(12,10), where degrees of freedom are 12 and 10

# To find Pr(X <= 3),
st.f.cdf(3,12,10) # gives 0.954299

# To find pdf f(3),
st.f.pdf(3,12,10) # gives 0.046852

# To find Pr(X > 3),
1 - st.f.cdf(3,12,10) # gives 0.0457007

# To find x such that Pr(X <= x) = 0.05,
st.f.ppf(0.05,12,10) # gives 0.363189

# To find x such that Pr(X >= x) = 0.05,
st.f.ppf(0.95,12,10) # gives 2.91298
```

## Displaying answers

The below function `disp` helps to print out all given arguments line by line, for sections with numerous values or answers.

```python
from st2334_helper import general as gn
"""Prints variable number of input arguments on separate lines.
Additionally, prints a blank line at the end. Mainly useful for
displaying numerous answers in the same function.

Parameters
----------
answers : any, varargs
    The answers to be printed out.
"""
a = True
b = 10
c = 0.5
gn.disp(a, b, c) # prints out True, 10, and 0.5 on separate lines

```

## Discrete pdf

For discrete p.d.f, calculation of the mean and variance requires the full set of data. The functions for computing both the mean and variance require a list of ordered pairs (given as tuples or lists).

Below is the function for calculating the mean, `find_mu`.

```python
from st2334_helper import general as gn
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
pdf = [(1, 0.1), (2, 0.2), (3, 0.2), (4, 0.3), (5, 0.2)]
mean = gn.find_mu(pdf) # gives 3.3
```

Below is the function for calculating the variance, `find_var`.

```python
from st2334_helper import general as gn
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
pdf = [(1, 0.1), (2, 0.2), (3, 0.2), (4, 0.3), (5, 0.2)]
var = gn.find_var(pdf) # gives 1.610
```

## Paired Data

Paired data may sometimes be given as a full two sets of data. In which case, `paired_data` can be used to compute the mean and variance of the paired data.

The input order must match for both lists.

```python
from st2334_helper import general as gn
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
x = [11.5, 11.7, 11.5, 11.8, 12.0, 12.2, 11.9]
y = [10.2, 10.3, 10.1, 10.6, 10.8, 11.3, 10.4]
mean, var = gn.paired_data(x, y) # gives 1.2714 and 0.03905
```

## Pooled Sample Variance

Sometimes there may be two different samples, but both with the same population variance, or maybe both from the same population. In such cases, we can use `pooled_sample_var` to compute the pooled variance for the samples.

```python
from st2334_helper import general as gn
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
n = [15, 20]
sample_var = [3.2, 4.1]
pooled_var = gn.pooled_sample_var(n, sample_var) # gives 3.71818
```

## Sum of Squares

The sum of squared differences with mean, is used primarily with the chi-squared distribution.

This sum of squares relate to sample variance through the following equation:

$$ \sum^n\_{i = 1} (X_i - \mu)^2 = (n - 1) S^2 $$

```python
from st2334_helper import general as gn
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
entry = [4.2, 4.5, 3.8, 4.0, 3.6, 3.9, 4.1]
mu = 4.0
ssq = gn.sum_squares(entry, mu) # gives 0.51
```

# Confidence Intervals

Computing confidence intervals is important. So important, that we have built a function for every case of building such intervals (or, _at least_, within the syllabus of our module).

It is recommended to first compute all required arguments, then proceed to use the correct function accordingly.

## Interval Bounds

With confidence intervals, there is a lower and upper bound to which we are confident of the population statistic.

Below is the function to calculate the upper bound, `upper_bound`.

```python
from st2334_helper import confidence_intervals as ci
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
upper = ci.upper_bound(0.95) # gives 0.975
```

Below is the function to calculate the lower bound, `lower_bound`.

```python
from st2334_helper import confidence_intervals as ci
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
lower = ci.lower_bound(0.95) # gives 0.025
```

## Minimum Size for a Desired Error

Sometimes we might have a particular amount of error that we find acceptable (or unacceptable), and we desire to achieve this level of error as we build our confidence intervals.

This error can be achieved by increasing the sample size, _n_, as the sample mean variance decreases until it converges to the population mean at infinity. We can scale this and find the minimum sample size, _n_ , for which a desired level of error is achieved.

Below is the function, `error_min_size`, which provides this calculation and informs us of the range of values that _n_ can take, at its minimum.

```python
from st2334_helper import confidence_intervals as ci
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
conf_level = 0.95
std = 1.5
err = 0.1
min_size = ci.error_min_size(conf_level, std, err)
# print "n >= 864.3282346561782"
```

# CI for Mean

Majority of the cases are regarding means. For such cases, most of the time we would be using either the _normal distribution_, or the _t-distrbution_.

The general rule of thumb is, if the population variance is known, then use the **normal distribution**. But if the population variance is unknown, then use the **t-distribution**. But then again, if the size is large (greater than or equal to 30), then the t-distribution approximates to the normal distribution, so we can use the **normal diistrubtion** with the sample variance.

This applies for not just single variable, but also double variable samples.
Sometimes we want to find the mean of difference between two populations, whether they are independent or not. In these cases, the rules on variance are the same as above, but additionally, if the two populations share the same variance, then a more accurate variance can be used by calculating a **pooled sample variance**.

## CI: Mean with Known Variance

Below is the function `mean_known`, for when population variance is known.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
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
mean = 4.5
conf_level = 0.95
n = 50
std = 1.5
ci = ci.mean_known(mean, conf_level, n, std)
gn.disp(ci)
# prints [4.084228852695096, 4.915771147304904]
```

## CI: Mean with Unknown Variance

Below is the function `mean_unknown` for when population variance is unknown.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
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
mean = 4.5
conf_level = 0.95
n = 50
sample_std = 1.5
ci = ci.mean_unknown(mean, conf_level, n, sample_std)
gn.disp(ci)
# prints [4.084228852695096, 4.915771147304904]
```

## CI: Difference in Mean with Known Variance

Below is the function `diff_known`, for when the population variance are both known.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
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
mean = [4.5, 3.0]
conf_level = 0.95
n = [30, 40]
var = [1.2, 0.8]
ci = ci.diff_known(mean, conf_level, n, var)
gn.disp(ci)
# prints [1.019908832364469, 1.980091167635531]
```

## CI: Difference in Mean with Unknown Variance

Below is the function `diff_unknown`, for when the population variance are unknown.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
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
mean = [4.5, 3.0]
conf_level = 0.95
n = [30, 40]
sample_var = [1.2, 0.8]
ci = ci.diff_unknown(mean, conf_level, n, sample_var)
gn.disp(ci)
# prints [1.0112119321670412, 1.9887880678329588]
```

## CI: Difference in Mean with Unknown but Equal Variance

Below is the function `diff_equal`, for when the population variance are unknown, but equal.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
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
mean = [4.5, 3.0]
conf_level = 0.95
n = [30, 40]
sample_var = [1.2, 0.8]
ci = ci.diff_equal(mean, conf_level, n, sample_var)
gn.disp(ci)
# prints [1.025188883082119, 1.974811116917881]
```

## CI: Difference in Mean of Paired Data

Below is the function `paired`, for when there is paired data with the mean and variance already computed.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
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
mean = 0.2833333333333334
conf_level = 0.95
n = 6
sample_var = 0.005666666666666632
ci = ci.paired(mean, conf_level, n, sample_var)
gn.disp(ci)
# prints [0.20433468825406964, 0.3623319784125971]
```

Below is the function `paired_raw`, for when the paired data is given as raw data, and the statistics are not yet computed. The data should then be given as two lists, with matching order and indexing.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
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
x = [4.5, 4.6, 4.3, 4.4, 4.7, 4.6]
y = [4.2, 4.2, 4.1, 4.1, 4.5, 4.3]
conf_level = 0.95
ci = ci.paired_raw(x, y, conf_level)
gn.disp(ci)
# prints [0.20433468825406964, 0.3623319784125971]
```

# CI for Variance

There are also confidence intervals for variance. These are fewer, and simpler to deal with.

For single variable cases, we only need to consider if the population mean is known or unknown. The distribution used is the same, and the only difference is in the use of the true population mean to compute a more accurate sum of squares, if it is known.

## CI: Variance with Known Mean

Below is the function `var_known`, for when the population mean is known. In this case, the set of data has to be given together with the population mean in order to calculate the sum of squares.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
"""Constructs a confidence interval for the variance of a population,
given that the population mean is known. The full data set has to be
provided to accurately compute.

Parameters
----------
entry : float[]
    The sample of data.

mu : float
    The population mean.

n : int
    The size of the sample.

conf_level : float
    The value of the confidence level.

Returns
-------
float[]
    The lower and upper bounds of the confidence interval respectively.
"""
entry = [4.3, 4.4, 4.6, 4.3, 4.7, 4.2, 4.4, 4.5, 4.6]
mu = 4.5
conf_level = 0.95
ci = ci.var_known(entry, mu, conf_level)
gn.disp(ci)
# prints [0.013142146434539969, 0.09257923718108749]
```

## CI: Variance with Unknown Mean

Below is the function `var_unknown`, for when the population mean is unknown. Then, the sample variance is used and should be given as input.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
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
sample_var = 1.2
n = 30
conf_level = 0.95
ci = ci.var_unknown(sample_var, n, conf_level)
gn.disp(ci)
# prints [0.7407526885916962, 2.0725669700949654]
```

## CI: Ratio of Variance

Below is the function `ratio_var`, for when the ratio of variance is to be used.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
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
sample_var = [3.5, 4.5]
n = [20, 30]
conf_level = 0.95
ci = ci.ratio_var(sample_var, n, conf_level)
gn.disp(ci)
# prints [0.3485801546361143, 1.86817764461591]
```

# Hypotheses Testing

The last section, hypotheses testing, is another major part of statistics (and particularly, _our module_).

For the sake of easier computation, in this package, we do not calculate the acceptance and rejection regions, nor the critical points. Instead, we find the **p-values** and compare them with the specified **level of significance** to reach a conclusion for the test.

Apart from just the tests, we also provide functionality for transforming from one distribution to another, and for calculating **p-value** directly.

## p-value

Sometimes when we compute p-value, we trouble ourselves as we think about whether the test is two-tailed, one-tailed, in which direction, and so on. We use these functions to bypass this, by letting it handle the logic behind it, as we pass the probability of an event, and indicate which scenario the test is.

Here, if the test is two-tailed, then we use a value of `0` to indicate this. If the test is one-tailed and testing if the statistic is less than the hypothesized value, then we use `-1` to indicate this. If it is testing if the statistic is more than, we use `1` to indicate this.

Below is the function `p_value`, used to return the p-value for a test.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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
p = 0.8
a = ht.p_value(p, 0) # for two-tailed test
b = ht.p_value(p, 1) # if more than in H_0
c = ht.p_value(p, -1) # if less than in H_0
gn.disp(a, b, c) # prints 0.39999, 1.9999, 0.8
```

Below is the function `pv_calc`, which computes the probability of the event given a distribution, and relevant statistics. This can be used for normal, t, and chi-squared distributions, but not the f-distribution.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
from scipy import stats as st
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
n = 30
mu = 5
var = 2
tails = 0

norm = ht.pv_calc(st.norm, n, mu, var, 4.8, tails) # test 4.8 against 5 for normal
t = ht.pv_calc(st.t, n, mu, var, 4.8, tails) # test 4.8 against 5 using t
chi = ht.pv_calc(st.chi2, n, mu, var, 2.3, tails) # test 2.3 against 2 using chi2
gn.disp(norm, t, chi) # prints 0.4385780260809994, 0.4448479915073278, 0.5276952687063057
```

Below is the function `pv_calc_f`, for calculating the p-value for an f-distribution.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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
n1, n2 = 24, 28
var1, var2 = 2.2, 2.4
sample_var1, sample_var2 = 2.4, 2.8
tails = 0
pv = ht.pv_calc_f(n1, n2, var1, var2, sample_var1, sample_var2, tails)
gn.disp(pv) # prints 0.5608233698758818
```

## Concluding the Test

Sometimes we may forget how to compare two numbers. Or maybe more realistically, forget how to properly conclude a hypothesis test after obtaining the p-value.

This function `comp_p_alpha` can help with that.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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
pv = 0.02
alpha = 0.05
gn.disp(ht.comp_p_alpha(pv, alpha)) # prints "0.02 < 0.05, H_0 is rejected"
```

# Transformers

Sometimes we need to transform certain statistics from one distribution to another. This is mostly done as intermediate workings, and in this package, the following transformers should be used largely by the final hypothesis test functions.
However, it is still definitely feasible to use them by themselves, though they require more caution.

## Binomial to Normal Transformer

Below is the function `binom_norm_transformer`, which approximates a binomial distribution to a normal distribution.

```python
from st2334_helper import hypotheses_tests as ht
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
n = 20
p = 0.4
transformer = ht.binom_norm_transformer(n, p)
```

## Normal to t Transformer

Below is the function `norm_t_transformer`, which transforms a normal distribution to a t distribution.

```python
from st2334_helper import hypotheses_tests as ht
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
n = 10
mu = 5
sample_var = 3
transformer = ht.norm_t_transformer(n, mu, sample_var)
```

## Difference in Mean to t Transformer

Below is the function `diff_t_transformer`, which transforms two normal distributions to a t distribution by taking the difference in mean.

```python
from st2334_helper import hypotheses_tests as ht
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
n = [15, 18]
mu = [4.5, 4.2]
var = [2.2, 2.4]
transformer = ht.diff_t_transformer(n, mu, var)
```

## Normal to Chi-squared Transformer

Below is the function `norm_chi2_transformer`, which transforms a normal distribution to a chi-squared distribution.

```python
from st2334_helper import hypotheses_tests as ht
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
n = 20
var = 4.2
transformer = ht.norm_chi2_transformer(n, var)
```

## Normal to f Transformer

Below is the function `norm_f_transformer`, which transforms two normal distributions to an f-distribution by taking the ratio of variance.

```python
from st2334_helper import hypotheses_tests as ht
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
var1 = 2.2
var2 = 2.4
transformer = ht.norm_f_transformer(var1, var2)
```

# Hypotheses Test for Mean

Majority of the cases are regarding means. For such cases, most of the time we would be using either the _normal distribution_, or the _t-distrbution_.

The general rule of thumb is, if the population variance is known, then use the **normal distribution**. But if the population variance is unknown, then use the **t-distribution**. But then again, if the size is large (greater than or equal to 30), then the t-distribution approximates to the normal distribution, so we can use the **normal diistrubtion** with the sample variance.

This applies for not just single variable, but also double variable samples.
Sometimes we want to find the mean of difference between two populations, whether they are independent or not. In these cases, the rules on variance are the same as above, but additionally, if the two populations share the same variance, then a more accurate variance can be used by calculating a **pooled sample variance**.

(Yes, this is the same as [Confidence Intervals](#confidence-intervals))

## HT: Mean with Known Variance

Below is the function `mean_known`, for when the population variance is known.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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
x_bar, mu, var, n = 4.5, 5, 1.5, 30
alpha = 0.05
tails = 0
ans = ht.mean_known(x_bar, mu, var, n, alpha, tails)
gn.disp(ans) # prints "0.025347318677468252 < 0.05, H_0 is rejected"
```

## HT: Mean with Unknown Variance

Below is the function `mean_unknown`, for when the population variance is unknown.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
"""Conducts a hypothesis test on the mean of a sample, given that
the population variance is unknown.

Parameters
----------
x_bar : float
    The mean of the sample.

mu : float
    The hypothesized mean of the population.

var : float
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
x_bar, mu, sample_var, n = 4.5, 5, 1.5, 30
alpha = 0.05
tails = 0
ans = ht.mean_unknown(x_bar, mu, sample_var, n, alpha, tails)
gn.disp(ans) # prints "0.025347318677468252 < 0.05, H_0 is rejected"
```

## HT: Difference in Mean with Known Variance

Below is the function `diff_known`, for when both population variance are known. If the hypothesized means are not individually known, just the relative difference in mean has to be represented correctly.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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
x_bar, mu, var, n = [5.0, 4.5], [5.2, 4.6], [1.2, 1.1], [24, 18]
alpha = 0.05
tails = 0
ans1 = ht.diff_known(x_bar, mu, var, n, alpha, tails)
mu2 = [0, 0]
ans2 = ht.diff_known(x_bar, mu2, var, n, alpha, tails)
gn.disp(ans1, ans2)
# prints "0.7641771556220935 > 0.05, H_0 is not rejected"
#        "0.13361440253771617 > 0.05, H_0 is not rejected"
```

## HT: Difference in Mean with Unknown Variance

Below is the function `diff_unknown`, for when the population variance are unknown. If the hypothesized means are not individually known, just the relative difference in mean has to be represented correctly

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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

var : float[]
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
x_bar, mu, sample_var, n = [5.0, 4.5], [5.2, 4.6], [1.2, 1.1], [24, 18]
alpha = 0.05
tails = 0
ans1 = ht.diff_unknown(x_bar, mu, sample_var, n, alpha, tails)
mu2 = [0, 0]
ans2 = ht.diff_unknown(x_bar, mu2, sample_var, n, alpha, tails)
gn.disp(ans1, ans2)
# prints "0.7657307167710179 > 0.05, H_0 is not rejected"
#        "0.141466776879148 > 0.05, H_0 is not rejected"
```

## HT: Difference in Mean with Unknown but Equal Variance

Below is the function `diff_equal`, for when the population variance are both unknown but equal. If the hypothesized means are not individually known, just the relative difference in mean has to be represented.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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

var : float[]
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
x_bar, mu, sample_var, n = [5.0, 4.5], [5.2, 4.6], [1.2, 1.1], [24, 18]
alpha = 0.05
tails = 0
ans1 = ht.diff_equal(x_bar, mu, sample_var, n, alpha, tails)
mu2 = [0, 0]
ans2 = ht.diff_equal(x_bar, mu2, sample_var, n, alpha, tails)
gn.disp(ans1, ans2)
# prints "0.7671725644521863 > 0.05, H_0 is not rejected"
#        "0.14394169373323762 > 0.05, H_0 is not rejected"
```

## HT: Difference in Mean of Paired Data

Below is the function `paired`, for when the mean and variance for paired data is already computed.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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
d_bar, mu_d, var, n = 0.283333333333333, 0, 0.005666666666666632, 6
alpha = 0.05
tails = 0
ans = ht.paired(d_bar, mu_d, var, n, alpha, tails)
gn.disp(ans) # prints "0.0002520662588245681 < 0.05, H_0 is rejected"
```

Below is the function `paired_raw`, for when only the raw set of paired data is available, and the mean and variance are not yet available.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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
x = [4.5, 4.6, 4.3, 4.4, 4.7, 4.6]
y = [4.2, 4.2, 4.1, 4.1, 4.5, 4.3]
mu_d = 0
alpha = 0.05
tails = 0
ans = ht.paired_raw(x, y, mu_d, alpha, tails)
gn.disp(ans) # prints "0.0002520662588245681 < 0.05, H_0 is rejected"
```

# Hypotheses Test for Variance

There are also hypotheses tests for variance. These are fewer, and simpler to deal with.

## HT: Variance

Below is the function `var_hypotest`. Can be used regardless of the population mean.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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
sample_var, var, n = 3.2, 3, 40
alpha = 0.05
tails = 0
ans = ht.var_hypotest(sample_var, var, n, alpha, tails)
gn.disp(ans) # prints "0.7164102211452501 > 0.05, H_0 is not rejected"
```

## HT: Ratio of Variance

Below is the function `var_ratio`, for comparing variance of two normal distributions. If the individual variance are not known, then only the relative ratio has to be represented correctly.

```python
from st2334_helper import general as gn
from st2334_helper import hypotheses_tests as ht
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
sample_var, var, n = [2.3, 2.6], [2.2, 2.5], [20, 25]
alpha = 0.05
tails = 0
ans1 = ht.var_ratio(sample_var, var, n, alpha, tails)
var2 = [1, 1] #if variance is not known
ans2 = ht.var_ratio(sample_var, var2, n, alpha, tails)
gn.disp(ans1, ans2)
# prints "0.5826098485356109 > 0.05, H_0 is not rejected"
#        "0.7938258588279136 > 0.05, H_0 is not rejected"
```
