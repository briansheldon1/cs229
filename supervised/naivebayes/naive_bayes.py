import random as rand
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.phi_0 = {}
        self.phi_1 = {}
        self.phi_y = {}
    
    def create_charset(self, X):
        self.charset = set()
        for x in X:
            for c in x:

                # ignore spaces
                if c==" ":
                    continue

            self.charset.add(c)

    def train(self, X_train: np.array, y_train: np.array):
        ''' Train using multinomial model '''
        
        # create set of characters to use
        self.create_charset(X_train)

        # get phi_spam (chance of spam)
        self.phi_y = np.mean(y_train)

        # initialize character counts to 1 for laplace smoothing
        phi_0 =  {c:1 for c in self.charset}
        phi_1 =  {c:1 for c in self.charset}

        # initialize denominators
        denom_0 = len(self.charset)
        denom_1 = len(self.charset)

        # iterate through examples, add character counts
        for x, y in zip(X_train, y_train):

            charcount = 0

            for c in x:

                # ignore if not in charset
                if c not in self.charset:
                    continue
                
                # record character seen
                charcount += 1
                if y==0:
                    phi_0[c] += 1
                else:
                    phi_1[c] += 1

            # record total characters seen
            if y==0:
                denom_0 += charcount
            else:
                denom_1 += charcount

        # divide character counts by num characters seen
        phi_0 = {c: v/denom_0 for c,v in phi_0.items()}
        phi_1 = {c: v/denom_1 for c,v in phi_1.items()}

        # store updated parameters
        self.phi_0 = phi_0
        self.phi_1 = phi_1

    def predict(self, X_test):
        
        preds = []
        for x in X_test:
            # start with probabilities of 1
            p0, p1 = 1, 1

            # get product of p(x_j|y=spam) and product of p(x_j|y=reg)
            for c in x:

                # ignore chars not in charset
                if c not in self.charset:
                    continue

                p0 += np.log(self.phi_0[c])
                p1 += np.log(self.phi_1[c])
            
            # multiply by p(y) (TESTING SWAPPED)
            p0 += np.log(1-self.phi_y)
            p1 += np.log(self.phi_y)

            pred = 1 if p1>p0 else 0
            preds.append(pred)

        return np.array(preds)

