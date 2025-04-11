import numpy as np

class SVM:
    ''' support vector machine implementation with SMO optimization '''

    def __init__(self, kernel="linear", tolerance=0.3):
        # store kernel function and tolerance
        self.kernel = kernel
        self.kernels = {
            "linear":     self.linear_kernel,
            "polynomial": self.polynomial_kernel,
            "gaussian":   self.gaussian_kernel
        }
        self.tolerance = tolerance

        # parameters to be trained
        self.alphas = np.array([])
        self.b = None

        # cache training data
        self.X_train = None
        self.y_train = None

    def train(self, X, y, C=0.01, iters_per_check=5, max_iters=100000):
        ''' train the svm model using simplified SMO '''

        m = len(X)
        alphas = [0 for i in range(m)]
        b = 0

        # compute kernel matrix
        K = self.create_kernel_matrix(X)
        if m > 1000:
            print("error: too large of m for kernel matrix")
            return

        for i in range(max_iters):
            preds = K @ alphas + b
            errors = preds - y

            # choose indices of alpha pairs to optimize
            if i % 10 == 0:
                r, s = np.random.choice(m, size=2, replace=False)
            else:
                r, s = self.select_rs(y, preds, alphas, C, errors)

            # optimize the two selected alphas
            ar_soln, as_soln = self.optimize_two_alphas(X, y, K, alphas, r, s, C)
            alphas[r] = ar_soln
            alphas[s] = as_soln

            # update b
            b = self.get_b(y, K, alphas)

            # stop if all constraints are satisfied
            if self.check_kkt(X, y, K, alphas, b, C):
                break
        else:
            print("warning: did not converge")

        self.alphas = alphas
        self.b = b
        self.X_train = X
        self.y_train = y

    def select_rs(self, y, preds, alphas, C, errors):
        ''' heuristic selection of alpha pair to optimize '''

        alphas = np.array(alphas)
        within = (alphas > 0) & (alphas < C)
        lower = (alphas == 0)
        upper = (alphas == C)

        cond1 = within & (errors == 0)
        cond2 = lower  & (y * errors >= 0)
        cond3 = upper  & (y * errors < 0)
        kkt_violate = cond1 & cond2 & cond3

        abs_errors = np.abs(errors)
        abs_errors[kkt_violate] = 0
        r = np.argmax(abs_errors)

        Er = errors[r]
        error_diffs = np.abs(errors - Er)
        s = np.random.choice(np.argpartition(error_diffs, -5)[-5:])

        return r, s

    def create_kernel_matrix(self, X):
        ''' compute full kernel matrix '''
        kernel_func = self.kernels[self.kernel]
        m = len(X)
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                K[i, j] = kernel_func(X[i], X[j])
        return K + K.T - np.diag(np.diag(K))

    def check_kkt(self, X, y, K, alphas, b, C):
        ''' check if all support vectors satisfy KKT conditions '''
        m = len(X)
        for i in range(m):
            kernel_vec = K[i, :]
            kernel_vec[i] = 0
            fm = y[i] * (sum(alphas * y * kernel_vec) + b)

            ai = alphas[i]
            e = self.tolerance
            if ai > 0 and ai < C:
                if fm > (1 + e) or fm < (1 - e):
                    return False
            elif ai == C:
                if fm > (1 + e):
                    return False
            elif ai == 0:
                if fm < (1 - e):
                    return False
        return True

    def optimize_two_alphas(self, X, y, K, alphas, r, s, C):
        ''' optimize pair of alphas subject to constraints '''

        yr, ys = y[r], y[s]
        y_alphas = y * alphas
        y_alphas[[r, s]] = 0
        gamma = -sum(y_alphas)

        if yr == ys:
            alpha_min = max(0, -yr * (ys * C - gamma))
            alpha_max = min(C, yr)
        else:
            alpha_min = max(0, yr * gamma)
            alpha_max = min(C, -yr * (ys * C - gamma))

        Kr = K[r, :]
        Kr[[r, s]] = 0
        Sr = yr * sum(y * alphas * Kr)

        Ks = K[s, :]
        Ks[[r, s]] = 0
        Ss = ys * sum(y * alphas * Ks)

        Krs = K[r, s]
        a = Krs
        b = 1 - yr * ys - Sr + yr * ys * Ss - gamma * yr * Krs

        ar_soln = self.solve_quadratic(a, b, [alpha_min, alpha_max])
        as_soln = (1 / ys) * (gamma - ar_soln * yr)

        return ar_soln, as_soln

    def solve_quadratic(self, a, b, bounds):
        ''' solve 1D quadratic maximization over bounds '''
        q = lambda x: a * x**2 + b * x
        candidates = [bounds[0], bounds[1]]
        if a != 0:
            x0 = -b / (2 * a)
            if bounds[0] < x0 < bounds[1]:
                candidates.append(x0)
        q_vals = [q(x) for x in candidates]
        return candidates[np.argmax(q_vals)]

    def get_b(self, y, K, alphas):
        ''' compute bias term b '''
        m = len(y)
        b_vec = y - K @ alphas
        return sum(b_vec) / m

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2, c=1, d=5):
        return (np.dot(x1, x2) + c) ** d

    def gaussian_kernel(self, x1, x2, sigma=0.1):
        return np.exp(-np.dot(x2 - x1, x2 - x1) / (2 * sigma ** 2))

    def predict(self, X_pred):
        ''' predict labels for new input points '''
        if self.X_train is None or self.y_train is None or self.alphas is None:
            print("model not trained")
            return

        kernel_func = self.kernels[self.kernel]
        preds = []
        for x_pred in X_pred:
            kernel_vec = np.array([kernel_func(x_pred, xi) for xi in self.X_train])
            sigmoid_arg = sum(self.alphas * self.y_train * kernel_vec) + self.b
            preds.append(1 if sigmoid_arg > 0 else -1)
        return np.array(preds)