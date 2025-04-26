import numpy as np

class PCA:
    def __init__(self):
        self.eigvecs = None

    def train(self, X, n):

        if n>=X.shape[1]:
            print("n must be less than X.size[1] (number of features of x_i)")
            return
        
        # normalize data
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)

        # get eigenvalues and eigenvectors
        sigma = X.T@X
        eigvals, eigvecs = np.linalg.eig(sigma)

        # get n highest eigenvalues
        top_eig_indices = np.argsort(eigvals)[-n:][::-1]
        self.eigvecs = eigvecs[top_eig_indices]

    def transform_x(self, X):
        return X@self.eigvecs.T

    def inv_transform(self, Y):
        ''' 
            Transform from u->x (transformed coords back into input)
            Note the transformation is an estimation and not exact
            x_i ~= y_1*u_1 + y_2*u_2 + ... + y_k*u_k
        '''
        return Y@self.eigvecs