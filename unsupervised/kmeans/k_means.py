import numpy as np

class KMeans:
    def __init__(self):
        """
        Initializes an empty KMeans model.
        """
        self.centroids = None

    def train(self, X: np.array, k=2, max_iters=10000, n_init=1):
        ''' 
        Train the KMeans model on input data X.

        Args:
            X (np.array): Input data (m, n) with m examples and n features.
            k (int): num of clusters/centroids to use in KMeans
            max_iters (int): Maximum number of iterations to update centroids.
            n_init (int): Number of times to initialize centroids 
                          (best chosen using self.loss(X))

        Notes:
            - Centroids are initialized using heuristic for spreading them out.
            - The algorithm stops when centroids converge or max_iters is reached.
        '''
        if k > X.shape[0]:
            print("k must be less than number of training examples")
            return
        
        best_centroids = None
        best_loss = None
        
        for n in range(n_init):

            # Initialize new centroids
            centroids = self._init_centroids(X, k)
            
            iters = 0
            while True:

                prev_centroids = centroids.copy()

                # Assign each point to the nearest centroid
                classes = []
                for x in X:
                    c_distances = [np.inner(x - c, x - c) for c in centroids]
                    classes.append(np.argmin(c_distances))

                # Update centroids based on current class assignments
                classes = np.array(classes)
                for c_index in range(len(centroids)):
                    X_class = X[classes == c_index]
                    if len(X_class) == 0:
                        centroids[c_index] = X[np.random.choice(len(X))]
                    else:
                        centroids[c_index] = sum(X_class) / len(X_class)

                # Check convergence or too many iterations
                iters += 1
                if np.allclose(prev_centroids, centroids) or iters >= max_iters:
                    break

                prev_centroids = centroids.copy()

            # check if soln beats best soln so far
            loss = self.loss(X, centroids)
            if best_loss is None or loss<best_loss:
                best_loss = loss
                best_centroids = centroids

        self.centroids = best_centroids

    def predict(self, X_pred, centroids=None):
        ''' 
        Predict the nearest cluster index for each example in X_pred.

        Args:
            X_pred (np.array): Data points to predict, shape (m, n).
            centroids : optional argument, if provided uses given centroids
                        instead of those stored in model (self.centroids)

        Returns:
            np.array: Predicted cluster indices for each input example.
        '''
        y_pred = []
        for x in X_pred:
            c_distances = [np.inner(x - c, x - c) for c in self.centroids]
            y_pred.append(np.argmin(c_distances))

        return np.array(y_pred)

    def _init_centroids(self, X, k):
        ''' 
        Initialize centroids using a heuristic to space them apart.

        Args:
            X (np.array): Dataset of shape (m, n).
            k (int):      Number of centroids to initialize.

        Returns:
            np.array: Initialized centroids of shape (k, n).
        '''
        m = X.shape[0]
        centroids = []
        centroid_dists = np.zeros((X.shape[0], k))
    
        for i in range(k):
            if i == 0:
                # Choose the first centroid randomly
                centroids.append(X[np.random.choice(m)])
            else:
                # Compute distances to nearest prior centroids
                prior_dists = centroid_dists[:, :i]
                min_dists = np.min(prior_dists, axis=1)
                min_dists = min_dists / sum(min_dists)
                centroids.append(X[np.random.choice(len(X), p=min_dists)])

            # Update distances for this new centroid
            for j in range(m):
                dist_vec = X[j] - centroids[i]
                centroid_dists[j, i] = np.inner(dist_vec, dist_vec)

        return np.array(centroids)
    

    def loss(self, X, centroids):
        ''' 
            Calculate kmeans loss for X given centroids 

            Args:
                X (np.array): Dataset of shape (m, n).
                centroids   : set of centroids of shape (k, n)
        '''
        
        distances = []
        for x in X:
            c_distances = [np.inner(x - c, x - c) for c in centroids]
            distances.append(min(c_distances))

        return sum(distances)