import numpy as np

class Node:
    def __init__(self):

        # pointers to l and r nodes
        self.left = None
        self.right = None
        
        # split is tuple of (var_index, split)
        self.split = None
        
        # label for only the final nodes
        self.label = None


class DecisionTree:
    def __init__(self, max_depth=4, min_leaf_size=5):
        
        # create root
        self.root = Node()
        
        # set max depth, min leaf size
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size

    
    def train(self, X, y, rand_features=False):
        
        self.split(self.root, X, y, rand_features=rand_features)

    def split(self, node: Node, X, y, depth=1, rand_features=False):

        # stopping conditions (min leaf, max depth)
        if len(X)<=self.min_leaf_size or \
                  depth>=self.max_depth or \
                    len(np.unique(y))==1:
            labels, counts = np.unique(y, return_counts=True)
            node.label = labels[np.argmax(counts)]
            return
        
        # store best split and min loss
        best_split = ()
        min_loss = None

        # get feature indices to iterate through (handle if random for random forest)
        if rand_features:
            n_features = len(X[0])
            sqrt_n = round(np.sqrt(n_features))
            f_indices = np.random.choice(n_features, sqrt_n, replace=False)
        else:
            f_indices = range(len(X[0]))


        for f in f_indices:

            # get sorted possible splits 
            splits = np.sort(X[:, f])

            for s_i in range(len(X)-1):

                # split is average between values
                split = (splits[s_i]+splits[s_i+1])/2

                # get cross entropy loss of split
                L = self.cross_entropy_split(X, y, f, split)

                # if better split found store it
                if min_loss is None or L<min_loss:
                    min_loss = L
                    best_split = (f, split)
        
        # store split
        node.split = best_split

        # create new left and right
        node.left = Node()
        node.right = Node()

        # get split data
        mask = X[:, best_split[0]] <= best_split[1]
        X_left, X_right = X[mask], X[~mask]
        y_left, y_right = y[mask], y[~mask]

        # recursively split
        self.split(node.left,  X_left,  y_left,  depth=depth+1)
        self.split(node.right, X_right, y_right, depth=depth+1)

    def cross_entropy_split(self, X, y, j, t):

        if len(X)==0:
            return 0
        
        mask = X[:, j]<=t
        y_left =  y[mask]
        y_right = y[~mask]

        # compute cross entropy
        L_left, L_right = 0, 0
        for c in set(y):
            if len(y_left)>0:
                pc_left = np.mean(y_left==c)
                if pc_left>0:
                    L_left -= pc_left*np.log2(pc_left)

            if len(y_right)>0:
                pc_right = np.mean(y_right==c)
                if pc_right>0:
                    L_right -= pc_right*np.log2(pc_right)
        
        # sum to get total
        L = (len(y_left)/len(y))*L_left + (len(y_right)/len(y))*L_right
        return L

    def predict(self, X):
        y = []
        for x in X:
            node = self.root
            while node.split is not None:
                j, t = node.split
                if x[j]<=t:
                    node = node.left
                else:
                    node = node.right

            y.append(node.label)

        return np.array(y)
        


        
