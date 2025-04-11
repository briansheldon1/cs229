import numpy as np

class MixGauss:
    def __init__(self):
        # Model parameters to be learned
        self.phi = None         # Mixture weights
        self.means = None       # Mean vectors for each component
        self.cov_mats = None    # Covariance matrices for each component

    def train(self, X: np.array, k, max_iters=10000, thresh=1e-7):
        """
        Train the Gaussian Mixture Model using the EM algorithm.

        Args:
        - X: Input data (m x n)
        - k: Number of clusters
        - max_iters: Maximum EM iterations
        - thresh: Convergence threshold on log-likelihood
        """
        m = X.shape[0]
        phi, means, cov_mats = self._init_params(X, k)
        cov_invs = [np.linalg.inv(cov) for cov in cov_mats]
        w = np.zeros((m, k))

        prev_loss = None
        curr_loss = None
        iters = 0

        while True:
            # ----- E-Step -----
            for i in range(m):
                for j in range(k):
                    w[i, j] = self.multivar_gauss(
                        X[i], means[j], cov_mats[j], cov_invs[j]
                    )
                w[i, :] /= np.sum(w[i, :])  # Normalize responsibilities

            # ----- M-Step -----
            for j in range(k):
                w_sum = np.sum(w[:, j])
                phi[j] = w_sum / m
                means[j] = X.T @ w[:, j] / w_sum

                X_minus_mean = X - means[j]
                cov_mats[j] = (
                    (w[:, j, np.newaxis] * X_minus_mean).T @ X_minus_mean / w_sum
                )
                cov_invs[j] = np.linalg.inv(cov_mats[j])

            # ----- Check for convergence -----
            prev_loss = curr_loss
            prob_X = [
                self.prob_x(x, phi, means, cov_mats, cov_invs) for x in X
            ]
            curr_loss = np.sum(np.log(prob_X))

            if prev_loss is not None and abs(curr_loss - prev_loss) < thresh:
                break

            iters += 1
            if iters >= max_iters:
                print("Warning: max iterations reached - did not converge")
                break

        # Store learned parameters
        self.phi = phi
        self.means = means
        self.cov_mats = cov_mats

    def _init_params(self, X, k):
        """Randomly initialize model parameters."""
        phi = np.ones((k,)) / k

        means_i = np.random.choice(len(X), size=k, replace=False)
        means = np.array([X[i] for i in means_i])

        n = X.shape[1]
        cov_mats = np.array([np.identity(n) for _ in range(k)])

        return phi, means, cov_mats

    def multivar_gauss(self, x, mu, cov_mat, cov_inv):
        """
        Compute the multivariate Gaussian pdf at x.
        """
        n = len(x)
        det_cov = np.linalg.det(cov_mat)
        norm_const = 1 / (np.sqrt((2 * np.pi) ** n * det_cov))
        diff = x - mu
        exp_part = np.exp(-0.5 * diff.T @ cov_inv @ diff)

        return norm_const * exp_part

    def create_sim_data(self, n_points):
        """
        Generate synthetic data based on learned distribution.
        """
        if self.phi is None or self.means is None or self.cov_mats is None:
            print("Model is untrained - cannot simulate data")
            return np.array([])

        X_sim = []
        for _ in range(n_points):
            j = np.random.choice(len(self.phi), p=self.phi)
            x = np.random.multivariate_normal(
                self.means[j], self.cov_mats[j]
            )
            X_sim.append(x)

        return np.array(X_sim)

    def prob_x(self, x, phi, means, cov_mats, cov_invs):
        """
        Compute the likelihood of point x under the current model.
        """
        k = len(phi)
        prob = 0
        for j in range(k):
            prob += (
                phi[j]
                * self.multivar_gauss(x, means[j], cov_mats[j], cov_invs[j])
            )
        return prob