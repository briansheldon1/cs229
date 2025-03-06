import numpy as np
from .decision_tree import DecisionTree

class AdaBoost:
    def __init__(self, max_depth=5, min_leaf_size=10):

        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size

        # prepare to store models and alpha values
        self.models: list[DecisionTree] = []
        self.alphas: list[float] = []

    def train(self, X, y, num_models=10):

        # initialize weights to 1/N
        N = len(X)
        w = np.ones(N)/N

        for m in range(num_models):

            # train model
            model = DecisionTree(max_depth=self.max_depth, 
                                 min_leaf_size=self.min_leaf_size)
            model.train(X, y, weights=w)

            # get training preds
            preds = model.predict(X)

            # get model error
            error = w.T@np.where(preds!=y, 1, 0)
            alpha = 0.5*np.log((1-error)/error)

            # store model and alpha
            self.models.append(model)
            self.alphas.append(alpha)

            # update weights
            log_L = -np.where(preds==y, 1, -1)
            w = w*np.exp(alpha*log_L)
            w *= (1/sum(w))

    def predict(self, X):
        
        # get model predictions (Gxm for G models, m training ex+)
        preds = np.array([np.where(Gm.predict(X)==0, -1, 1) for Gm in self.models])

        # multiply and sum by weights alpha 
        preds = preds.T@np.array(self.alphas) #(mxG @ Gx1 = mx1)
        preds = np.where(preds>0, 1, 0)

        return preds


