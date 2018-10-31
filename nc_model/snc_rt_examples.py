import numpy as np
import matplotlib.pyplot as plt

'''
    This example is being used to implement the example calculations 
    from the "Performance Modelling and Analysis of Unreliable Links with Retransmissions using Network Calculus"
    paper. We will implement the calculation of the model and plot the results for different parameter configurations
    
    @Author: Alexander Mildner
    @Email: mildner@in.tum.de
'''

'''
Notes for myself: 
    [x]+ -> max{x,0}
    beta -> beta (N) -> beta_param[N]

'''

# init values for paper example (paper#1)
vp = 0.1
r = 0.1
R = 1
T = 3
b = 3
epsilon = 0.001
W = 8
N = 2
C = vp
B = 1.0 - epsilon
vp_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_arr = [1, 2, 3]

'''
# init values for paper example (paper#2)
vp = 0.1
r = 0.5
R = 0.8
T = 3
b = 3
epsilon = 0.01
W = 8
N = 2
C = 0.1250
B = 5.8750
'''

def alpha(x, a, c):
    y = (a*x)+c
    return y


def beta(x, a, c):
    y = a*max(0, x - c)
    return y


def max_plus(value):
    return max(0, value)


def ci_power(i):
    return C**i


def ci_power_i(i):
    return i*(C**i)


def summation(N, i, f):
    res = 0
    for j in xrange(i, N+1):
        res += f(j)
    return res


# Case for N = N
def calculate_phi():
    phi_res = np.zeros(N)

    for it in xrange(0, N):
        double_sum = 0
        for p in xrange(it, N):
            inner = 0
            for q in xrange(0, p+1):
                inner += ci_power(q)
            double_sum += inner

        phi_res[it] = R*T + (b*summation(N, it+1, ci_power)) + B*double_sum + (r*W*summation(N, it+1, ci_power_i))

    return phi_res


def calculate_matrix():
    s = (N, N)
    A = np.zeros(s)

    for row in xrange(0, N):
        for col in xrange(0, N):
            if row == col:
                A[row][col] = R - 2.0*r*summation(N, row+1, ci_power)
            else:
                A[row][col] = -r*summation(N, col+1, ci_power)
    return A


def calculate_a_i(matrix, i, phi):
    if i == 0:
        raise NameError("i must be > 0")
    res = matrix.copy()
    for row in xrange(0, N):
        for col in xrange(0, N):
            if col == i-1:
                res[row][col] = phi[row]
    return res


def calculate_t_i_approx(matrix, i, phi):
    t_i = calculate_a_i(matrix, i, phi)

    sign_a, log_det_a = np.linalg.slogdet(matrix)
    sign_a_i, log_det_a_i = np.linalg.slogdet(t_i)

    det_a = sign_a * np.exp(log_det_a)

    det_a_i = sign_a_i * np.exp(log_det_a_i)

    res = det_a_i / det_a
    return res


def calculate_b_j(matrix, j):
    t_tmp = 0
    c_tmp = 0
    for i in xrange(1, j+1):
        t_tmp += calculate_t_i_approx(matrix, i, phi)
    for k in xrange(0, j):
        c_tmp += C**k
    result = ((C**j)*r)*t_tmp + (C**j)*b + c_tmp*B + j*(C**j)*r*W
    return result


# Step by Step Calculation example
phi = calculate_phi()
print "Phi: \n"
print phi
print "\n"

matrix_a = calculate_matrix()
print "Matrix A: \n"
print matrix_a
print "\n"

'''
matrix_a_n = calculate_a_i(matrix_a, 2, phi)
print "Matrix A_N: \n"
print matrix_a_n
print "\n"

t_i_approx = calculate_t_i_approx(matrix_a, 2, phi)
print "Ti,approx : \n"
print t_i_approx
print "\n"
'''

t = np.arange(0, 21)
alpha_param = []
colors_a = ['k*-', 'r--', 'b--', 'g--', 'y--']

# Calculate parameters of alpha curves (Arrival Curves) & plot them
for k in xrange(0, N+1):
    b_j = calculate_b_j(matrix_a, k)
    t_i_calc = calculate_t_i_approx(matrix_a, k+1, phi)
    r_alpha = (C**k)*r
    alpha_param.append([r_alpha, b_j])

beta_param = []
beta_param.append([R, T])
R_tmp = R
T_tmp = T

# Calculate parameters of beta curves (Service Curves)
for k, it in reversed(list(enumerate(alpha_param))):
    if k == 0:
        break
    R_tmp -= it[0]
    T_tmp += it[1]
    beta_param.insert(0, [R_tmp, T_tmp])

print "Alpha Parameter: \n"
print alpha_param
print "\n"
print "Beta Parameter: \n"
print beta_param

colors_b = ['black', 'r-.', 'b-.', 'g-.', 'y-.']

# Plot alpha Curves (Arrival Curves)
for k, it in list(enumerate(alpha_param)):
    plt.plot(t,
             it[0] * t + it[1],
             colors_a[k],
             label="Arrival Curve (" + str(k) + ") r=" + str(it[0]) + " b=" + str(round(it[1], 2)) + "")

# Plot beta curves (Service curves)
for k, it in list(enumerate(beta_param)):
    plt.plot(t,
             it[0]*t-it[1],
             colors_b[k],
             label="Left-over Service Curve (" + str(k) + ") R=" + str(it[0]) + " T=" + str(round(it[1], 2)) +"")

# Calculate aggregated Arrival Curve
tmp_r = 0
tmp_b = 0
for k, it in list(enumerate(alpha_param)):
    tmp_r += it[0]
    tmp_b += it[1]

agg_alpha = [tmp_r, tmp_b]
print agg_alpha
# Calculate delay bounds
delay_bounds = []
for k, it in list(enumerate(alpha_param)):
    y_int = it[0] * 0 + it[1]
    result = (y_int + beta_param[k][0]*beta_param[k][1]) / beta_param[k][0]
    delay_bounds.append(result)
    plt.axhline(y=y_int,
                xmax=result/20,
                label="Delay Bound ("+str(k)+") d <= " + str(round(result, 2)),
                color='c')

print "Delay Bounds: \n"
print delay_bounds

# Recall delay bound calculation
y_int_re = agg_alpha[0] * 0 + agg_alpha[1]
result = (y_int_re + beta_param[N][0]*beta_param[N][1] / beta_param[N][0])

print "Delay Bound for aggregated Flow: \n"
print result
# Plot Settings
plt.ylim(0, 20)
plt.xlim(0, 20)
plt.xlabel("Time")
plt.ylabel("Data")
plt.legend(loc='upper left')
plt.grid(True)
plt.figure(figsize=(3.45, 4.0))
plt.show()

