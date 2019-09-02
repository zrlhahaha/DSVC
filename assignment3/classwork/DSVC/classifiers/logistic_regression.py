import numpy as np
import random
import math

class LogisticRegression(object):

    def __init__(self):
        self.w = None           #logistic回归theta
        self.ww = None          #softmax回归theta
        self.www = None         #多类别logistic回归theta

    # def sigmoid(self,x):
    #     return 1/(1+np.exp(-x))

    def cost(self,x,y):
        m = x.shape[0]                              
        theta = self.w.reshape(-1,1)
        y = y.reshape(-1,1)                         #theta[n,1]  y[m,1]  x[m,n]
        h = self.sigmoid(x.dot(theta))               #h[m,1]
        print(np.sum((1-h) == np.sum(np.zeros_like(h))))
        print(np.sum((h) == np.sum(np.zeros_like(h))))
        cost = -y.T.dot(np.log(h))-(1-y).T.dot(np.log(1-h))     
        return cost[0][0]/m

    def gradient(self,x,y):
        m = x.shape[0]                          
        theta = self.w.reshape(-1,1)        
        y = y.reshape(-1,1)                 #x[m,n]  y[m,1] theta[n,1]
        h = self.sigmoid(x.dot(theta))            #h[100,1]         
        res = x.T.dot(h - y)/m              #res[3,1]
        return res.flatten()


    def sigmoid(self,z):
            return 1.0 / ( 1.0 + np.exp(-z) )

    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        loss = self.cost(X_batch,y_batch)
        grad = self.gradient(X_batch,y_batch)
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        return loss,grad

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            index = np.random.choice(num_train, batch_size)
            X_batch = X[index]
            y_batch = y[index]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w += -learning_rate*grad
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            #if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            

        return loss_history


    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        theta = self.w.reshape(-1,1)
        h = self.sigmoid(X.dot(theta))
        for i in range(X.shape[0]):
            if(h[i]>=0.5):
                y_pred[i] = 1             
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose = True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        """

        num_train,dims = X.shape

        if self.ww is None:
            self.ww = 0.001 * np.random.randn(10,dims)

        for i in range(10):
            y_train = np.zeros(num_train)
            for n in range(num_train):
                if(y[n] == i):
                    y_train[n] = 1

            self.train(X,y_train,learning_rate,num_iters,batch_size,verbose)
            print("-----------------------")
            self.ww[i] = self.w
            
    def one_vs_all_predict(self,X):
        num_test,dims = X.shape
        y_point = np.zeros((10,num_test),float)
        for i in range(10):
            theta = self.ww[i].reshape(-1,1)
            y_point[i] = self.sigmoid(X.dot(theta)).flatten()

        y_pred = np.zeros(num_test)

        for i in range(num_test):
            y_pred[i]
            y_pred[i] = np.argmax(y_point[:,i].flatten())

        return y_pred
            
    def argmax(self,X):
        max = -100000000
        maxIndex = -1
        for i in range(len(X)):
            if(X[i]>max):
                maxIndex=i
                max = X[i]

        return maxIndex 
            
            
        
            


        

    # def softmax(self,z):
    #     return np.exp(z) / np.sum(np.exp(z))

    # def sotfmax_loss(self,X,y):
    #     loss = 0
    #     num_train,dims = X.shape
    #     for i in range(num_train):
    #         xi = X[i].reshape(dims,1)
    #         loss += self.softmax(self.ww.T.dot(xi))[y[i]]
        
    #     return loss / num_train


    # def softmax_train(self, X, y, learning_rate=1e-3, num_iters=100,
    #         batch_size=200, verbose = True):
    #     train_num,dims = X.shape
    #     loss_history = []
                                                                              
    #     if(self.ww == None):
    #         self.ww = 0.001 * np.random.randn(dims,10)               #X[m,n]  Xi.reshape[n,1]  ww[n,k]  z[k,1]

        
    #     for it in range(num_iters):
    #         grad_i = np.zeros_like(self.ww)
    #         index = np.random.choice(train_num, batch_size)                     #point[k,1]  y_vector[k,1]
    #         X_batch = X[index]
    #         y_batch = y[index]

    #         for i in range(batch_size):
    #             Xi = X_batch[i].reshape(-1,1)
    #             yi = y_batch[i]
    #             y_vector = np.zeros((10,1),float)                  
    #             y_vector[yi] = 1
    #             z = self.ww.T.dot(Xi)
    #             point = self.softmax(z)
    #             grad_i += Xi * (point - y_vector).T

    #         grad_i = grad_i / batch_size
    #         self.ww = self.ww + learning_rate * grad_i

    #         loss = self.sotfmax_loss(X_batch,y_batch)
    #         loss_history.append(loss)
    #         if verbose and it%10 == 0:
    #             print("iter:",it," loss:",loss)

    #     return loss_history

    # def softmax_predict(self,X):
    #     p = []
    #     for Xi in X:
    #         point = self.softmax(self.ww.T.dot(Xi.reshape(-1,1)))
    #         p.append(point.argmax())

    #     return p

     



