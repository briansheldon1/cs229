from pca import PCA
import numpy as np

X = np.array([np.random.randn(5) for i in range(15)])

model = PCA()
model.train(X, 2)

Y = model.transform_x(X)
X_recovered = model.inv_transform(Y)