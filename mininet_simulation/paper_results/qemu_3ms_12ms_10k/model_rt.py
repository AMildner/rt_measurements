import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

"""@package docstring
RTModel

    Model Class for SNC framework to calculate delay/backlog bounds 
    for Unreliable Links.
    
    Further Information:
    "Performance Modelling and Analysis of Unreliable Links with Retransmissions using Network Calculus"
    
    @Author: Alexander Mildner
    @Email: mildner@in.tum.de
"""


class RTModel:
    """RTModel

    This Class initializes the SNC model for UL with RTs. The input data list object holds all the necessary
    parameters for the calculations.

    :param data:    This list contains the input parameters in the following scheme [p, r, R, T, b, epsilon, W, N]

    """

    def __init__(self, data):
        if len(data) is not 8:
            raise NameError("Data length is: " + str(len(data)) + " but must be 8!\n")
        # Init values
        self.p = data[0]
        self.r = data[1]
        self.R = data[2]
        self.T = data[3]
        self.b = data[4]
        self.epsilon = data[5]
        self.W = data[6]
        self.N = data[7]
        self.C = self.p
        self.B = 1.0 - self.epsilon
        self.phi = []

        # Return values
        self.A = np.zeros((self.N, self.N))
        self.alpha_params = []
        self.beta_params = []
        self.agg_alpha = []
        self.delay_bounds = []
        self.backlog_bounds = []
        self.delay_bound = 0.0
        self.backlog_bound = 0.0

        self.calculate()

    # Private Functions

    def __summation(self, n, i,  func):
        res = 0
        for j in xrange(i, n + 1):
            res += func(j)
        return res

    def __ci_power(self, i):
        return self.C ** i

    def __ci_power_i(self, i):
        return i*(self.C ** i)

    def __calculate_matrix_a(self):
        """__calculate_matrix_a

            This method calculates the matrix A
        """
        s = (self.N, self.N)
        result = np.zeros(s)

        for row in xrange(0, self.N):
            for col in xrange(0, self.N):
                if row == col:
                    result[row][col] = self.R - 2.0 * self.r * self.__summation(self.N, row + 1, self.__ci_power)
                else:
                    result[row][col] = -self.r * self.__summation(self.N, col + 1, self.__ci_power)

        self.A = result

    def __calculate_phi(self):
        """__calculate_phi

            This method calculates the phi vector
        """
        phi_res = np.zeros(self.N)

        for it in xrange(0, self.N):
            double_sum = 0
            for p in xrange(it, self.N):
                inner = 0
                for q in xrange(0, p + 1):
                    inner += self.__ci_power(q)
                double_sum += inner

            phi_res[it] = self.R * self.T + (self.b * self.__summation(self.N, it + 1, self.__ci_power)) + \
                          self.B * double_sum + \
                          (self.r * self.W * self.__summation(self.N, it + 1, self.__ci_power_i))

        self.phi = phi_res

    def __calculate_a_i(self, i):
        """__calculate_a_i

        This function calculates the adapted Matrix A(i) for the current iteration i

        :param i: number of iteration
        :return: A(i)
        """
        if i == 0:
            raise NameError("i must be > 0")
        res = self.A.copy()
        for row in xrange(0, self.N):
            for col in xrange(0, self.N):
                if col == i - 1:
                    res[row][col] = self.phi[row]
        return res

    def __calculate_t_i_approx(self, i):
        """__calculate_t_i_approx

        This function calculates the T(i,inf) for the current iteration i

        :param i: number of iteration
        :return: T(i,inf)
        """
        t_i = self.__calculate_a_i(i)

        sign_a, log_det_a = np.linalg.slogdet(self.A)
        sign_a_i, log_det_a_i = np.linalg.slogdet(t_i)

        det_a = sign_a * np.exp(log_det_a)
        det_a_i = sign_a_i * np.exp(log_det_a_i)

        res = det_a_i / det_a
        return res

    def __calculate_b_j(self, j):
        """__calculate_b_j

        This function calculates the b(j,inf) values for each iteration of the arrival curves

        :param j: number of iteration
        :return: returns b(j,inf)
        """
        t_tmp = 0
        c_tmp = 0
        for i in xrange(1, j + 1):
            t_tmp += self.__calculate_t_i_approx(i)
        for k in xrange(0, j):
            c_tmp += self.C ** k
        result = ((self.C ** j) * self.r) * t_tmp + (self.C ** j) * \
                                                    self.b + c_tmp * self.B + j * (self.C ** j) * self.r * self.W
        return result

    def __calculate_r_alpha(self, i):
        """__calculate_r_alpha

        This function calculates the rate of the current iteration of the arrival curves

        :param i: number of iteration
        :return: returns rate
        """
        return (self.C**i)*self.r

    def __calculate_alpha_params(self):
        # Calculate parameters of alpha curves (Arrival Curves)
        for k in xrange(0, self.N + 1):
            b_j = self.__calculate_b_j(k)
            r_alpha = (self.C ** k) * self.r
            self.alpha_params.append([r_alpha, b_j])

    def __calculate_beta_params(self):
        self.beta_params.append([self.R, self.T])
        R_tmp = self.R
        T_tmp = self.T

        # Calculate parameters of beta curves (Service Curves)
        for k, it in reversed(list(enumerate(self.alpha_params))):
            if k == 0:
                break
            R_tmp -= it[0]
            T_tmp += it[1]
            self.beta_params.insert(0, [R_tmp, T_tmp])

    def __calculate_agg_alpha_params(self):
        tmp_r = 0
        tmp_b = 0
        for k, it in list(enumerate(self.alpha_params)):
            tmp_r += it[0]
            tmp_b += it[1]
        self.agg_alpha = [tmp_r, tmp_b]

    def calculate_delay_bounds(self):
        for n in xrange(0, self.N+1):
            y_int_re = self.alpha_params[n][0] * 0 + self.alpha_params[n][1]
            delay_bound = ((y_int_re + (self.beta_params[n][0] * self.beta_params[n][1])) /
                                self.beta_params[n][0])
            self.delay_bounds.append([y_int_re, delay_bound])
        return self.delay_bounds

    def calculate_backlog_bounds(self):
        for n in xrange(0, self.N+1):
            x_int_bk = self.beta_params[n][1]
            # x_int_bk = 0 - self.beta_params[n][0] * self.beta_params[n][1] / self.beta_params[n][0]
            backlog_bound = self.alpha_params[n][0] * x_int_bk + self.alpha_params[n][1]
            self.backlog_bounds.append([x_int_bk, backlog_bound])
        return self.backlog_bounds

    def calculate(self):
        """

        This function calculates all the necessary Values for getting the following return parameters:
            alpha_params : List of Parameters for the Arrival Curves
            beta_params : List of Parameters for the Service Curve
            delay_bound : Delay Bound for the given set of input parameters
            backlog_bound : Backlog Bound for the given set of input parameters

        :return: Calculates the according Return Parameters of the Object
        """
        # Reset Return Lists
        if len(self.alpha_params) is not 0 and len(self.beta_params) is not 0:
            self.alpha_params = []
            self.beta_params = []

        # Recalculate Matrix A and Vector PHI
        self.__calculate_matrix_a()
        self.__calculate_phi()

        # Calculate alpha and beta params
        self.__calculate_alpha_params()
        self.__calculate_beta_params()

        # Calculate aggregate alpha
        self.__calculate_agg_alpha_params()

        # Calculate aggregated delay Bound
        y_int_re = self.agg_alpha[0] * 0 + self.agg_alpha[1]
        self.delay_bound = ((y_int_re + (self.beta_params[self.N][0] * self.beta_params[self.N][1])) / self.beta_params[self.N][0])

        # Calculate aggregated backlog Bound
        x_int_bk = self.beta_params[self.N][1]
        # x_int_bk = 0 - self.beta_params[self.N][0] * self.beta_params[self.N][1] / self.beta_params[self.N][0]
        self.backlog_bound = self.agg_alpha[0] * x_int_bk + self.agg_alpha[1]

    def get_alpha_params(self):
        """
        Returns a list of the parameters b,r of the arrival curves
        The list has length N+1 and the form [[b(0),r(0)],[b(1),r(1)],...,[b(N),r(N)]]

        :return: List of parameters for the arrival curves
        """

        return self.alpha_params

    def get_beta_params(self):
        """
        Returns a list of the parameters b,r of the arrival curves
        The list has length N+1 and the form [[R(0),T(0)],[R(1),T(1)],...,[R(N),T(N)]]

        :return: List of parameters for the arrival curves
        """

        return self.beta_params

    def set_p_param(self, i):
        self.C = i
        self.p = i

    def set_epsilon_param(self, i):
        self.epsilon = i
        self.B = 1 - i

    def check_stability(self):
        if (self.R > ((1-self.C**(float(self.N+1)))/(1-self.C))*self.r):
            return True
        else:
            return False


""" Main Method
    Example Execution:
"""
if __name__ == '__main__':

    # data : [p, r, R, T, b, epsilon, W, N]
    data = [0.1, 0.1, 1, 30, 3, 0.001, 120, 3]
    data_1 = [0.1, 0.1, 1, 3, 3, 0.001, 8, 3]

    model = RTModel(data)
    if (model.check_stability()):
        print "OK"
    else:
        print "Not OK"

    # Plot the first graph of the paper (Data vs. Time)
    t = np.arange(0.0, 21.0, 0.1)
    colors_css = [name for name, hsv in mcolors.CSS4_COLORS.iteritems()]
    colors_a = ['k-', 'r--', 'b--', 'g--', 'y--']
    colors_b = ['black', 'r-.', 'b-.', 'g-.', 'y-.']
    # Plot alpha Curves (Arrival Curves)
    db = model.calculate_delay_bounds()
    bb = model.calculate_backlog_bounds()
    for k, it in list(enumerate(model.alpha_params)):
        plt.plot(t,
                 it[0] * t + it[1],
                 colors_a[k],
                 label="Arrival Curve (" + str(k) + ") r=" + str(it[0]) + " b=" + str(round(it[1], 2)) + "")
    # Plot beta curves (Service curves)
    # Added max function -> increased resolution of time step size (1->0.1)
    # Otherwise the plot starts at different T value
    max_val = []
    for k, it in list(enumerate(model.beta_params)):
        for j in t:
            max_val.append(it[0] * max(0, (j - it[1])))
        plt.plot(t,
                 max_val,
                 colors_b[k],
                 label="Left-over Service Curve (" + str(k) + ") R=" + str(it[0]) + " T=" + str(round(it[1], 2)) + "")
        max_val = []

    for n in xrange(0, model.N+1):
        plt.hlines(db[n][0], 0, db[n][1], 'brown', alpha=0.7)

    for n in xrange(0, model.N+1):
        plt.vlines(bb[n][0], 0, bb[n][1], 'grey', alpha=0.7)

    print "Values: \n"
    print model.backlog_bound
    print model.delay_bound

    plt.ylim(0, 21)
    plt.xlim(0, 21)
    plt.xlabel("Time")
    plt.ylabel("Data")
    plt.legend(loc='upper left', prop={'size': 9})
    #plt.grid(True)
    plt.show()
    #plt.savefig("paper_graph1_final.png", bbox_inches="tight")
    plt.close()

    # Plot the 2nd graph of the paper (Delay Bound vs. Loss Probability)
    loss_p = np.arange(0.1, 1.0, 0.1)

    delay_bounds = []
    for i in xrange(1, 5):
        data[7] = i
        tmp_delay = []
        for k, it in list(enumerate(loss_p)):
            # Recalculate the model with the according parameters
            tmp_model = RTModel(data)
            tmp_model.set_p_param(it)
            tmp_model.calculate()
            # Test, adding maximum feedback delays to the delay bounds
            tmp_delay.append(tmp_model.delay_bound + float(i+1)*float(tmp_model.W))
        delay_bounds.append(tmp_delay)

    colors_delay = ['r--', 'b-', 'g-.', 'k:']
    for k, it in list(enumerate(delay_bounds)):
        plt.plot(loss_p, it, colors_delay[k], label=str(k+1) + " Retransmission Flow(s)")

    plt.ylim(0, 800)
    plt.xlim(0.1, 0.9)
    plt.xlabel("Loss Probability")
    plt.ylabel("Delay Bound")
    plt.legend(loc='upper left')
    #plt.grid(True)
    plt.show()
    #plt.savefig("paper_graph2_final.png", bbox_inches="tight")
    plt.close()

    # Plot the 3rd graph of the paper (Delay bound vs. Maximum Feddback Delay)
    max_feed = np.arange(0, 41, 1)

    delay_bounds = []
    for i in xrange(1, 5):
        data[7] = i
        tmp_delay = []
        for k, it in list(enumerate(max_feed)):
            # Recalculate the model with the according parameters
            tmp_model = RTModel(data)
            tmp_model.set_p_param(0.7)
            tmp_model.W = it
            tmp_model.calculate()
            tmp_delay.append(tmp_model.delay_bound)
        delay_bounds.append(tmp_delay)

    colors_delay = ['r--', 'b-', 'g-.', 'k:']
    for k, it in list(enumerate(delay_bounds)):
        plt.plot(max_feed, it, colors_delay[k], label=str(k + 1) + " Retransmission Flow(s)")

    plt.ylim(0, 36)
    plt.xlim(0, 40)
    plt.xlabel("Maximum Feedback Delay")
    plt.ylabel("Delay Bound")
    plt.legend(loc='upper left')
    #plt.grid(True)
    plt.show()
    #plt.savefig("paper_graph3_final.png", bbox_inches="tight")
    plt.close()
    '''
    # Test out different input parameters
    loss_p = np.arange(0.1, 1.0, 0.1)

    delay_bounds = []
    for i in xrange(1, 4):
        data[7] = i
        tmp_delay = []
        for k, it in list(enumerate(loss_p)):
            # Recalculate the model with the according parameters
            tmp_model = RTModel(data)
            tmp_model.set_p_param(it)
            tmp_model.calculate()
            tmp_delay.append(tmp_model.delay_bound)
        delay_bounds.append(tmp_delay)

    colors_delay = ['r--', 'b-', 'g-.']
    for k, it in list(enumerate(delay_bounds)):
        plt.plot(loss_p, it)

    plt.ylim(0, 47)
    plt.xlim(0.1, 0.9)
    plt.xlabel("Loss Probability")
    plt.ylabel("Delay Bound")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    '''
