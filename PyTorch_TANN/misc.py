import numpy as np
import torch

class preprocessing():
    def __init__(self):
        return
    
    class EarlyStopper:
        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = np.inf

        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
    
    def shuffle(self,x,rnd):
        return x[rnd]

    def slice_data(self,x,ntrain,nval):
        return [x[:(ntrain+nval)],x[(ntrain+nval):]]

    def GetParams(self,x,norm=True,vectorial_norm=False):
        '''
        Compute normalization parameters:
            - normalize ([-1,1]) component by component (vectorial_norm = True)
            - normalize ([-1,1]) (vectortial_norm = False, norm = True)
            - standardize (vectorial_norm = False, norm = False)

            x: data
                :type array
            vectorial_norm: normalize data component by component (along axis = 1)
                :type boolena
            norm: normalization [-1,1]
                :type boolean
        '''
        if vectorial_norm == False:
            if norm == True:
                # normalize [-1,1]
                A = 0.5*(np.amax(x)-np.amin(x)); B = 0.5*(np.amax(x)+np.amin(x))
            else:
                # standardize (mean = 0, std = 1)
                A = np.std(x,axis=0); B = np.mean(x,axis=0);
                if A.shape[0]>1: A[A==0]=1
        else:
            # normalize component by component (along axis = 1)
            dim = x.shape[-1]
            u_max = np.zeros((dim,))
            u_min = np.zeros((dim,))
            for i in np.arange(dim):
                u_max[i] = np.amax(x[:,i])
                u_min[i] = np.amin(x[:,i])
            A = (u_max-u_min)/2.
            B = (u_max+u_min)/2.
            A[A==0]=1
        return torch.tensor(np.array([np.float64(A),np.float64(B)]))
    
    def Normalize(self,inputs,prm):
        '''
        Normalize features
        :inputs : data
        :prm : normalization parameters
        '''
        return np.divide(np.add(inputs, -prm[1]), prm[0])
    
    def DeNormalize(self,outputs,prm):
        '''
        Denormalize features
        :output : dimensionless data
        :prm : normalization parameters
        '''
        return np.add(np.multiply(outputs, prm[0]), prm[1])