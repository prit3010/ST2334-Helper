from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
from st2334_helper import hypotheses_tests as ht
from scipy import stats as st

mean = 0.2833333333333334
sample_var = 0.005666666666666632
d_bar, mu_d, var, n = 20.283333333333333, 0, 0.005666666666666632, 6
alpha = 0.05
tails = 0
ans = ht.paired(d_bar, mu_d, var, n, alpha, tails)
gn.disp(ans)
