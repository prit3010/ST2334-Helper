from functions import *

# Questions below

def q1():
    v1 = 100 * 0.2 * 0.8
    v2 = 16
    v3 = 12
    v4 = 16
    ans = (v1 + v2 + v3 + v4) / 16
    disp(ans)

def q2():
    ans = True
    disp(ans)

def q3():
    ans = 0.5
    disp(ans)

def q4():
    ans = True
    disp(ans)

def q5():
    ans = 4
    disp(ans)

def q6():
    ans = False
    disp(ans)

def q7():
    point = (35 * 20) / 5.12**2
    ans = st.chi2.cdf(point, 35)
    disp(ans)

def q8():
    n = [25, 36]
    std = [1.88, 1.43]
    var = pooled_sample_var(n, std)
    ans = var / n[0] + var / n[1]
    disp(ans)

def q9():
    n = 49
    xbar = 567
    sd = 35
    ans = ci_known(xbar, 0.95, n, sd)
    disp(ans)

def q10():
    x = [31, 32, 28, 27]
    s = [3, 9, 6, 4]
    ans = []
    for i in range(4):
        trans = norm_to_t_transformer(36, x[i], s[i]**2)
        ans.append(st.t.cdf(trans(30), 36-1))
    disp(ans)

# Generate solutions below

q1()
q2()
q3()
q4()
q5()
q6()
q7()
q8()
q9()
q10()
