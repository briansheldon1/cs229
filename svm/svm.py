import numpy as np

class SVM:
    def __init__(self, kernel="linear", tolerance=0.3):
        
        # store kernel function
        self.kernel = kernel
        self.kernels = {
            "linear":     self.linear_kernel,
            "polynomial": self.polynomial_kernel,
            "gaussian"  : self.gaussian_kernel
        }
        
        # store tolerance
        self.tolerance = tolerance

        # initialize empty parameters to train
        self.alphas = np.array([])
        self.b = None

        # other parameters used for train/pred
        self.X_train = None
        self.y_train = None

    def train(self, X, y, C=0.01, iters_per_check=5, max_iters=100000):
  
        # num training examples
        m = len(X)

        # Initialize alphas to 0 and b to 0
        alphas = [0 for i in range(m)]
        b = 0

        # create kernel matrix of K_ij = K(x_i, x_j)
        K = self.create_kernel_matrix(X)

        # Collect kernel matrix
        if m>1000:
            print("Error: too large of m for kernel matrix")
            return

    
        for i in range(max_iters):

            # update b and current training predictions
            #  (training preds used in error calculations and heuristics)
            
            preds = K@alphas + b

            # get errors on each alpha (how much in violation of KKT)
            errors = preds - y

            # grab two random alpha indices (CHANGE TO HEURISTICS LATER)
            if i%10==0:
                r,s = np.random.choice(m, size=2, replace=False)
            else:
                r, s = self.select_rs(y, preds, alphas, C, errors)
            
            # get updated alpha values
            ar_soln, as_soln = self.optimize_two_alphas(X, y, K, alphas, r, s, C)
            
            # store new alphas
            alphas[r] = ar_soln
            alphas[s] = as_soln
            
            # update b
            b = self.get_b(y, K, alphas)

            # check passing kkt conditions, break)
            if self.check_kkt(X, y, K, alphas, b, C):
                break
        
        else:
            print(f"Warning: did not converge")
        
        # store parameters for predictions
        self.alphas = alphas
        self.b = b
        self.X_train = X
        self.y_train = y
  
    def select_rs(self, y, preds, alphas, C, errors):

        # select r as point with largest error that violates kkt
        kkt_violate = np.zeros_like(preds)

        alphas = np.array(alphas)

        # get mask of kkt violations (below represent violations)
        #   cond1: 0<alpha<C then error==0
        #   cond2: alpha==0  then y*error >= 0
        #   cond3: alpha==C  then y*error < 0
        within = (alphas>0) & (alphas < C)
        lower = (alphas==0)
        upper = (alphas==C)

        cond1 = within & (errors == 0)
        cond2 = lower  & (y*errors >= 0)
        cond3 = upper  & (y*errors < 0)

        kkt_violate = (cond1) & (cond2) & (cond3)

        # get r (location of maximum |error| with kkt violate)
        abs_errors = np.absolute(errors)
        abs_errors[kkt_violate] = 0
        r = np.argmax(abs_errors)

        # get s (maximize |Er - Es|)
        Er = errors[r]
        error_diffs = np.absolute(errors - Er)
        s = np.random.choice(np.argpartition(error_diffs, -5)[-5:])

        return r, s
    
    def create_kernel_matrix(self, X):

        kernel_func = self.kernels[self.kernel]
        m = len(X)

        K = np.zeros((m,m))
        for i in range(m):
            for j in range(i, m):
                K[i, j] = kernel_func(X[i], X[j])

        K = K+K.T - np.diag(np.diag(K))
        return K

    def check_kkt(self, X, y, K, alphas, b, C):

        m = len(X)
        
        for i in range(m):

            # get kernel vector for i'th support vector
            kernel_vec = K[i, :]
            kernel_vec[i] = 0
            
            # get functional margin
            fm = y[i]*(sum(alphas*y*kernel_vec) + b)

            # check if breaking KKT, if so return early False
            ai = alphas[i]
            e = self.tolerance
            if ai>0 and ai<C:
                if fm>(1+e) or fm<(1-e):
                    return False
            elif ai==C:
                if fm>(1+e):
                    return False
            elif ai==0:
                if fm<(1-e):
                    return False
        
        # if all points pass then return True
        return True
    
    def optimize_two_alphas(self, X, y, K, alphas, r, s, C):

        # get y's of index1 and index2
        # note (r=index1, s=index2)
        yr = y[r]
        ys = y[s]

        # get gamma (-sum of i=!{r,s} of alpha_i*y_i)
        y_alphas = y*alphas
        y_alphas[[r,s]] = 0
        gamma = -sum(y_alphas)

        # get alpha_r bounds
        if yr==ys:
            alpha_min = max(0, -yr*(ys*C-gamma))
            alpha_max = min(C, yr)
            alpha_bounds = [alpha_min, alpha_max]

        else:
            alpha_min = max(0, yr*gamma)
            alpha_max = min(C, -yr*(ys*C-gamma))
            alpha_bounds = [alpha_min, alpha_max]

        # get sums over kernels
        #    Sr = yr*(sum over i!=r,s y_i*a_i*K(x_r, x_i))
        #    Ss = ys*(sum over i!=r,s y_i*a_i*K(x_s, x_i))
        Kr = K[r, :]
        Kr[[r, s]] = 0
        Sr = yr*sum(y*alphas*Kr)

        Ks = K[s, :]
        Ks[[r, s]] = 0
        Ss = ys*sum(y*alphas*Ks)

        Krs = K[r, s]

        # collect quadratic a, b and c
        a = Krs
        b = 1-yr*ys-Sr+yr*ys*Ss-gamma*yr*Krs

        # get optimal value
        ar_soln = self.solve_quadratic(a, b, alpha_bounds)
        as_soln = (1/ys)*(gamma-ar_soln*yr)

        return ar_soln, as_soln

    def solve_quadratic(self, a, b, bounds):

        # function for getting quadratic point
        q = lambda x: a*x**2 + b*x

        # get possible solns (endpoints are always possible)
        candidates = [bounds[0], bounds[1]]

        # if quadratic term, include deriv=0 as candidate
        if a!=0:
            x0 = -b/(2*a)
            if x0>bounds[0] and x0<bounds[1]:
                candidates.append(x0)
        
        # otherwise compare endpoints
        q_vals = [q(x) for x in candidates]
        return candidates[np.argmax(q_vals)]
 
    def get_b(self, y, K, alphas):
        m = len(y)
        b_vec = y - K@alphas
        return sum(b_vec)/m

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    def polynomial_kernel(self, x1, x2, c=1, d=5):
        return (np.dot(x1, x2)+c)**d
    
    def gaussian_kernel(self, x1, x2, sigma=0.1):
        return np.exp(-np.dot(x2-x1, x2-x1)/(2*sigma**2))
    
    def predict(self, X_pred):
        if self.X_train is None or \
            self.y_train is None or \
             self.alphas is None:
            print("model not trained")
            return
        
        kernel_func = self.kernels[self.kernel]
        
        preds = []
        for x_pred in X_pred:
            
            # calculate dot products between each training sample
            kernel_vec = np.array([kernel_func(x_pred, xi) for xi in self.X_train])

            sigmoid_arg = sum(self.alphas*self.y_train*kernel_vec)+self.b
            if sigmoid_arg>0:
                preds.append(1)
            else:
                preds.append(-1)

        return np.array(preds)
