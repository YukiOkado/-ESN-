import numpy as np
import matplotlib.pyplot as plt

class Esn:
    def __init__(self,M,Ntrain,Ntest,g=0.9,eps=0.1):
        self.M=M
        self.Ntrain = Ntrain
        self.Ntest = Ntest
        self.u = self.input_signal()
        self.ytag1, self.ytag2 = self.target()
        self.xmat_train, self.xmat_test = self.reserve()
        self.weight = self.cal_weight()
        self.y_train = self.output()
        #self.X = X
        #self.Y = Y
        #self.y = y
        

    def input_signal(self):
        data = np.loadtxt("C:/Users/oo/Documents/santafe.txt")
        self.u = data[0:self.Ntrain+self.Ntest+1]
        return self.u
        
    def target(self):
        self.y_tag1=self.u[1:self.Ntrain+1]
        self.y_tag2=self.u[self.Ntrain+1:self.Ntrain+self.Ntest+1]
        
        return self.y_tag1, self.y_tag2

    def reserve(self,g=0.9,eps=0.1):
        Jseed = 11
        np.random.seed(int(Jseed))
        J=np.random.normal(0,np.sqrt(1/self.M),(self.M,self.M))
        x=np.zeros((self.M,1))
        _xmat = x
        for i in range(self.Ntrain+self.Ntest):
            x = g * (np.dot(J,x) + eps * ((self.u[i]-3)/252 * 2 - 1))
            x = np.tanh(x)

            _xmat = np.append(_xmat,x,axis=1)

        self.xmat_train = _xmat[:,1:self.Ntrain+1]
        self.xmat_test = _xmat[:,self.Ntrain+1:self.Ntrain+self.Ntest+1]

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

    def aka(self,X , Y ,g=0.9,eps=0.1):
        Jseed = 11
        #xx= self.xmat_train[:,self.Ntrain-1:self.Ntrain]
        #yy=self.y_train[-1]
        np.random.seed(int(Jseed))
        J = np.random.normal(0,np.sqrt(1/self.M),(self.M,self.M))
        X = g * (np.dot(J,X) + eps * ((Y-3)/252 * 2 - 1))
        X       = np.tanh(X)

        return X

    def ao(self, XX ,YY):
        YY = XX.T@self.weight

        return YY

    def ki(self, g=0.9,eps=0.1):
        x = self.xmat_train[:,-1]
        
        y = self.y_train[-1]
        y_test = y
        for i in range(self.Ntest):
            x = self.aka(x,y)
            y = self.ao(x,y)

            y_test = np.append(y_test,y)

        y_test = y_test[1:]

        return y_test

    def gosa(self ,ki):
        MSE = np.sum((self.y_tag2-self.ki()) ** 2)/self.Ntest

        return MSE
for k in [10,20,30, 40 ,100]:
    esn = Esn(1000,2000,k)
    esn.input_signal()
    hai, hoi = esn.target()
    esn.reserve()
    esn.cal_weight()
    esn.output()
    ro=esn.ki()
    hei =esn.gosa(ro)
    print(hei)
    plt.plot(hoi,label ="Target Signal")
    plt.plot(ro, label ="prerdict")
    plt.legend()
    plt.show()


