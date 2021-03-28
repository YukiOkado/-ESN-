import numpy as np
from math import *
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

class Reservior_computing_BAmodel:
    def __init__(self, Number_Nodes, Number_Train,
                 Number_Test, Nonlinearity = 0.9,
                 Input_Sensitivity = 0.1):

        self.Number_Nodes = Number_Nodes
        self.Number_Train = Number_Train
        self.Number_Test = Number_Test
        self.u = self.input_signal()
        self.ytag1, self.ytag2 = self.target_signal()
        self.xmat_train, self.xmat_test = self.reservior_training()
        self.weight = self.cal_weight()
        self.y_train = self.output()

    def input_signal(self):
        data = np.loadtxt("C:/Users/oo/Documents/santafe.txt")
        self.u = data[0:self.Number_Train+self.Number_Test+1]
        return self.u
        
    def target_signal(self):
        self.y_tag1=self.u[1:self.Number_Train+1]
        self.y_tag2=self.u[self.Number_Train+1:
                           self.Number_Train+self.Number_Test+1]
        
        return self.y_tag1, self.y_tag2

    def reservior_training(self, Nonlinearity = 0.9, Input_Sensitivity=0.1):
        Jseed = 11
        np.random.seed(int(Jseed))
        P=np.random.normal(0,np.sqrt(1/self.Number_Nodes),
                           (self.Number_Nodes,self.Number_Nodes))
  
        G=nx.barabasi_albert_graph(self.Number_Nodes,4,seed=11)
        J=nx.adjacency_matrix(G)
        J=J.toarray()
        Q=np.multiply(P,J)
        l=np.linalg.eig(Q)
        la=max(abs(l[0]))
        Q=Q/la
        degree_hist = nx.degree_histogram(G) 
        degree_hist = np.array(degree_hist, dtype=float)
        degree_prob = degree_hist/G.number_of_nodes()
        plt.loglog(np.arange(degree_prob.shape[0]),degree_prob,"b.")

        plt.xlabel('k')
        plt.ylabel('p(k)')
        plt.title('Degree Distribution')
        yoko=np.arange(degree_prob.shape[0])[np.nonzero(degree_prob)]
        tate=degree_prob[np.nonzero(degree_prob)]
        print(max(G.degree(), key = lambda x: x[1])) 
    
        dayo=np.polyfit(np.log(yoko),np.log(tate),1)
   
        origin=dayo[0]*np.log(yoko)+dayo[1]
    
        print(r2_score(origin,np.log(tate)))
    
        plt.show() 
        xx=np.zeros((self.Number_Nodes,1))
        _xmat = xx
        for i in range(self.Number_Train+self.Number_Test):
            xx = Nonlinearity * (np.dot(J,xx) + Input_Sensitivity * ((self.u[i]-3)/252 * 2 - 1))
            xx = np.tanh(xx)

            _xmat = np.append(_xmat,xx,axis=1)

        self.xmat_train = _xmat[:,1:self.Number_Train+1]
        self.xmat_test = _xmat[:,self.Number_Train+1:
                               self.Number_Train+self.Number_Test+1]

        return self.xmat_train,self.xmat_test

    def cal_weight(self):
        bmat = self.xmat_train@self.y_tag1
        amat = self.xmat_train@self.xmat_train.T
        amat_inv = np.linalg.pinv(amat)
        self.weight = amat_inv@bmat
        return self.weight

    def output(self):
        self.y_train = self.xmat_train.T@self.weight

        return self.y_train

    def reservior_prediction(self,X , Y ,Nonlinearity = 0.9, Input_Sensitivity =0.1):
        Jseed = 11
        np.random.seed(int(Jseed))


        P = np.random.normal(0,np.sqrt(1/self.Number_Nodes),
                             (self.Number_Nodes,self.Number_Nodes))
        G = nx.barabasi_albert_graph(self.Number_Nodes,4,seed=11)
   
        J = nx.adjacency_matrix(G)
        J = J.toarray()
        Q = np.multiply(P,J)
        l = np.linalg.eig(Q)
        la = max(abs(l[0]))
        Q  =  Q/la
        X = Nonlinearity * (np.dot(Q,X) 
                            + Input_Sensitivity * ((Y-3)/252 * 2 - 1))
        X       = np.tanh(X)

        return X

    def output_prediction(self, XX ,YY):
        YY = XX.T@self.weight

        return YY

    def Prediction(self, Nonlinearity =0.9, Input_Sensitivity =0.1):
        x = self.xmat_train[:,-1]
        
        y = self.y_train[-1]
        y_test = y
        for i in range(self.Number_Test):
            x = self.reservior_prediction(x,y)
            y = self.output_prediction(x,y)

            y_test = np.append(y_test,y)

        y_test = y_test[1:]

        return y_test

    def Evaluation(self ,ki):
        MAE = np.sum(abs(self.y_tag2-self.Prediction()))/self.Number_Test

        return MAE
   
esn = Reservior_computing_BAmodel(1000,2000,10)
esn.input_signal()
target1, target2 = esn.target_signal()
esn.reservior_training()
esn.cal_weight
esn.output()
prediction=esn.Prediction()
result =esn.Evaluation(prediction)
print(result)
plt.plot(target2,label ="Target Signal")
plt.plot(prediction, label ="prerdict")
plt.legend()
plt.show()

