from .decision_tree import DecisionTree
import numpy as np
class RandomForest:
    def __init__(self, max_depth=5, min_leaf_size=10):
        self.models = None

        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size

    def train(self, X, y, num_models=10):

        # get bootstrap samples
        Z_features, Z_labels = self.bootstrap(X, y, num_models=num_models)

        # train model for each Z
        models: list[DecisionTree] = []
        for i in range(num_models):
            G = DecisionTree(max_depth=self.max_depth, min_leaf_size=self.min_leaf_size)
            G.train(Z_features[i], Z_labels[i])
            models.append(G)

        # record models for prediction
        self.models = models
    
    def bootstrap(self, X, y, num_models=10):

        m = len(X)
        Z_features = []
        Z_labels = []
        
        # create Z sampled from X, y
        for i in range(num_models):

            # get randomly selected indices of X
            Z_indices = np.random.randint(m, size=(m,))

            # get random X and y samples
            Z_X = X[Z_indices]
            Z_y = y[Z_indices]

            # record total new bootstrap sample
            Z_features.append(Z_X)
            Z_labels.append(Z_y)

        return Z_features, Z_labels
    
    def predict(self, X):

        def max_label(row):
            values, counts = np.unique(row, return_counts=True)
            return values[np.argmax(counts)]

        preds = np.array([G.predict(X) for G in self.models])
        preds = np.apply_along_axis(func1d=max_label, axis=0, arr=preds)

        return preds