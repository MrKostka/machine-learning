import numpy as np
import random
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, recall_score, precision_score,f1_score, jaccard_score
from matplotlib import pyplot as plt

#SVM class with all methods and SMO algorithm
class SVM():
    def __init__(self, reg_param_C=7.5, tol=0.0, max_passes = 300):
        self.reg_param_C = reg_param_C
        self.tol = tol
        self.max_passes = max_passes

    '''
    This method calculate predicted value y for unknown vector (entry) X
    according to decision rule, e.i. decision boundary expression
    '''
    def calculate_f(self, actual_index, X, y, alphas, m, b):
        sum = 0
        for i in range(0, m):
            sum += np.dot(X[i], X[actual_index]) * y[i] * alphas[i]
        return  sum + b

    '''
    This method calculate value of L according to equality of ys
    Later it returns maximum from 2 values
    '''
    def calculate_L(self, i, j, y, alphas):
        return max(0, alphas[j]-alphas[i]) if y[i] != y[j] else max(0, alphas[i] + alphas[j] - self.reg_param_C )

    '''
    This method calculate value of H according to equality of ys
    Later it returns minimum from 2 values
    '''
    def calculate_H(self, i, j, y, alphas):
        return min(self.reg_param_C, self.reg_param_C + alphas[j] - alphas[i]) if y[i] != y[j] else min( self.reg_param_C , alphas[i] + alphas[j] )

    '''
    This method calculate value of eta according to 2 X vectors, returned value
    represents subtraction between their dot product (multiplied by 2) and dot products of
    themself alone
    '''
    def calculate_eta(self, i, j, X):
        return 2*np.dot(X[i],X[j]) - np.dot(X[i],X[i]) - np.dot(X[j],X[j])

    '''
    Update of alpha_j according Errors i and j, y_j and eta, later clip it in interval [L;H] according to clipping rule
    '''
    def update_alpha_j(self, j, alphas, E_i, E_j, eta, L, H):
        alpha_j = alphas[j] - ( ( E_i - E_j ) * y[j] ) / eta
        if alpha_j > H:
            alpha_j = H
        if alpha_j < L:
            alpha_j = L
        return alpha_j

    '''
    Update of aplha_i according to y_i and y_j, old_alpha_j and actual alpha_j
    '''
    def update_alpha_i(self, alphas, y, old_alpha_j, i, j):
        return alphas[i] + y[i] * y[j] * (old_alpha_j - alphas[j])


    '''
    Calculation of b1
    '''
    def calculate_b1(self, i, j, b, E_i, X,  y, alphas, old_alpha_i, old_alpha_j):
        return b - E_i - y[i]*(alphas[i] - old_alpha_i)*np.dot(X[i], X[i]) - y[j]*(alphas[j] - old_alpha_j)*np.dot(X[i],X[j])

    '''
    Calculation of b2
    '''
    def calculate_b2(self, i, j, b, E_j, X,  y, alphas, old_alpha_i, old_alpha_j):
        return b - E_j - y[i]*(alphas[i] - old_alpha_i)*np.dot(X[i], X[j]) - y[j]*(alphas[j] - old_alpha_j)*np.dot(X[j],X[j])

    '''
    Update of bias according to alpha_i and alpha_j and b1 and b2
    '''
    def update_b(self, i,j, alphas, b1, b2):
        if alphas[i] > 0 and alphas[i] < self.reg_param_C:
            return b1
        if alphas[j] > 0 and alphas[j] < self.reg_param_C:
            return b2
        return (b1 + b2) / 2


    def train(self, X, y):
        m = len ( X )
        alphas = np.zeros( m )
        E = np.zeros ( m )
        self.b = 0
        passes = 0
        #here we go with SMO - sequantial minimal optimization
        while(passes < self.max_passes):
            print("No. of passes: " + str(passes))
            alphas_changed = 0
            for i in range( 0, m):
                E[i] = self.calculate_f(i, X, y, alphas, m, self.b) - y[i]
                if ( (y[i]*E[i] < -self.tol) and (alphas[i] < self.reg_param_C) ) or (
                        (y[i]*E[i] > self.tol) and (alphas[i] > 0) ):
                    j = random.choice( [ k for k in range(0,m) if k != i ] )
                    E[j] = self.calculate_f(j, X, y, alphas, m, self.b) - y[j]
                    old_alpha_i = alphas[i]
                    old_alpha_j = alphas[j]
                    L = self.calculate_L(i, j, y, alphas)
                    H = self.calculate_H(i, j, y, alphas)
                    if L == H:
                        continue
                    eta = self.calculate_eta(i, j, X)
                    if eta >= 0:
                        continue
                    alphas[j] = self.update_alpha_j(j, alphas, E[i], E[j], eta, L, H)
                    if abs( alphas[j] - old_alpha_j ) < 0.00001:
                        continue
                    alphas[i] = self.update_alpha_i( alphas, y, old_alpha_j, i, j )
                    b1 = self.calculate_b1(i, j, self.b, E[i], X,  y, alphas, old_alpha_i, old_alpha_j)
                    b2 = self.calculate_b2(i, j, self.b, E[j], X,  y, alphas, old_alpha_i, old_alpha_j)
                    self.b = self.update_b(i,j, alphas, b1, b2)
                    alphas_changed += 1
            if alphas_changed == 0:
                passes += 1
            else:
                passes = 0

        self.w = np.zeros(X[0].shape)
        for i in range(0, len(X) ):
            self.w += alphas[i] * y[i] * X[i]
        print("*-----------------------------------------------------------------*")

    '''
    Returns actual w vector
    '''
    def get_weight_vector(self):
        return self.w

    '''
    Returns actual bias
    '''
    def get_bias(self):
        return self.b

    def predict(self, X, y):
        y_pred = np.zeros(len(y))
        for i in range (0, len(X)):
            if ( np.dot( self.w, X[i] ) + self.b ) >= 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
            print("X:   " + str(X[i]) + "   y_true:     " + str(y[i]) + "   y_predicted:    " + str(y_pred[i]))

        print("*-----------------------------------------------------------------*")
        print("Precision score: " + str( precision_score(y, y_pred)))
        print("Recall score: " + str( recall_score(y, y_pred)))
        print("F1_score score: " + str( f1_score(y, y_pred)))
        print("Jaccard score: " + str( jaccard_score(y, y_pred)))
        print("Confusion matrix: ")
        print(confusion_matrix(y, y_pred))
        print("*-----------------------------------------------------------------*")



#Data importing and preprocessing
X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.8)
y[y == 0] = -1
tmp = np.ones(len(X))
y = tmp * y

#Data splitted on training and testing sets
X_train, y_train, X_test, y_test = X[:int( len(X)*0.8 )], y[:int( len(y)*0.8 )], X[int( len(X)*0.8 ):], y[int( len(y)*0.8 ):]

#Visualization
X_train_pos, X_train_neg, X_test_pos, X_test_neg = [], [], [], []
for l in range(0, len(X)):
    if l < len(X_train):
        if y_train[l] == 1:
            X_train_pos.append( X_train[l] )
        else:
            X_train_neg.append( X_train[l] )
    if l < len(X_test):
        if y_test[l] == 1:
            X_test_pos.append( X_test[l] )
        else:
            X_test_neg.append( X_test[l] )
tr_pos = plt.scatter(np.array(X_train_pos)[:, 0], np.array(X_train_pos)[:, 1], c='greenyellow', cmap='winter')
tr_neg = plt.scatter(np.array(X_train_neg)[:, 0], np.array(X_train_neg)[:, 1], c='pink', cmap='winter')
te_pos = plt.scatter(np.array(X_test_pos)[:, 0], np.array(X_test_pos)[:, 1], c='green', cmap='winter')
te_neg = plt.scatter(np.array(X_test_neg)[:, 0], np.array(X_test_neg)[:, 1], c='red', cmap='winter')
plt.legend((tr_pos, tr_neg, te_pos, te_neg), ('Train positive','Train negative','Test positive','Test negative'), scatterpoints=1, loc='upper left')

#Model object creation - too high C influences possible overfitting
model = SVM(reg_param_C=7.5, tol=0.0, max_passes = 50)
model.train(X_train, y_train)
print("Weight vector: " + str(model.get_weight_vector()))
print("Bias: " + str(model.get_bias()))
print("*-----------------------------------------------------------------*")
model.predict(X_test, y_test)
w = model.get_weight_vector()
b = model.get_bias()

#Lines visualization
def f(x, w, b, c=0):
    return (-w[0] * x - b + c) / w[1]

# w.x+ b = 0
a0 = -4; a1 = f(a0, w, b)
b0 = 4; b1 = f(b0, w, b)
plt.plot([a0,b0], [a1,b1], 'k')

# w.x + b = 1
a0 = -4; a1 = f(a0, w, b, 1.0)
b0 = 4; b1 = f(b0, w, b, 1.0)
plt.plot([a0,b0], [a1,b1], 'k--')

# w.x + b = -1
a0 = -4; a1 = f(a0, w, b, -1.0)
b0 = 4; b1 = f(b0, w, b, -1.0)
plt.plot([a0,b0], [a1,b1], 'k--')

plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.title("SVM with SMO on linear separable data")
plt.show()