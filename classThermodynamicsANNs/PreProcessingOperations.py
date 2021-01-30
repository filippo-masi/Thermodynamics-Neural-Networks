'''
Created on 26 Jan 2021

@author: filippomasi & ioannisstefanou
'''

import numpy as np
import pickle

class preProcessing():
    def __init__(self,file_inp="",file_out="",silent=True):
        self.silent=silent
        if file_inp == "" or file_out == "":
            if self.silent==False: print("No data provided for training")
        else:
            if self.silent==False: print("... Loading data")
            with open(file_inp, 'rb') as f_obj:
                inputs = pickle.load(f_obj)    
            with open(file_out, 'rb') as f_obj:
                outputs = pickle.load(f_obj)
            e_t,e_tdt,De_t,ap_t,sg_t = inputs
            ap_tdt,Dap_t,F_tdt,sg_tdt,Dsg_t,D_tdt = outputs
            self.n_samples = e_tdt.shape[0]
            self.train_percentage=.5; self.n_train_samples = int(round(self.n_samples * self.train_percentage)); self.n_val_samples = int(round(self.n_train_samples * (1.- self.train_percentage))); self.n_test_samples = self.n_val_samples
            if self.silent==False: print("... Slice data into training, validation, and test data-sets")
            self.nrm_inp,self.nrm_out,inputs,outputs = self.slice_TVTt(e_t,e_tdt,De_t,ap_t,sg_t,ap_tdt,Dap_t,F_tdt,sg_tdt,Dsg_t,D_tdt)
            self.uN_T,self.uN_V,self.uN_Tt,self.oN_T,self.oN_V,self.oN_Tt = self.preInandOut(inputs,outputs)
            
        
    def slice_TVTt(self,e_t,e_tdt,De_t,ap_t,sg_t,ap_tdt,Dap_t,F_tdt,sg_tdt,Dsg_t,D_tdt):
        '''Slice data into training, validation, and test data-sets '''
        ε_t_tv=e_t[:(self.n_train_samples + self.n_val_samples)] # ε @ t
        Dε_t_tv=De_t[:(self.n_train_samples + self.n_val_samples)] # Δε @ t
        σ_t_tv=sg_t[:(self.n_train_samples + self.n_val_samples)] # σ @ t
        α_t_tv=ap_t[:(self.n_train_samples + self.n_val_samples)] # α @ t
        # Inputs (test) = ε_t, Δε_t, σ_t, α_t
        ε_t_test=e_t[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # ε @ t
        Dε_t_test=De_t[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # Δε @ t
        σ_t_test=sg_t[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # σ @ t
        α_t_test=ap_t[(self.n_train_samples+self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # α @ t
        # Outputs (training + validation = tv) = F_(t+dt), α_(t+dt), σ_(t+dt)
        F_tdt_tv=F_tdt[:(self.n_train_samples+self.n_val_samples)] # F_(t+dt)
        Dα_tdt_tv=Dap_t[:(self.n_train_samples+self.n_val_samples)] # α_(t+dt)
        Dσ_tdt_tv=Dsg_t[:(self.n_train_samples+self.n_val_samples)] # σ_(t+dt)
        α_tdt_tv=ap_tdt[:(self.n_train_samples+self.n_val_samples)] # α_(t+dt)
        σ_tdt_tv=sg_tdt[:(self.n_train_samples+self.n_val_samples)] # σ_(t+dt)
        D_tdt_tv=D_tdt[:(self.n_train_samples+self.n_val_samples)] # D_(t+dt)
        # Outputs (test) = F_(t+dt), α_(t+dt), σ_(t+dt)
        F_tdt_test=F_tdt[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # F_(t+dt)
        Dα_tdt_test=Dap_t[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # α_(t+dt)
        Dσ_tdt_test=Dsg_t[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # σ_(t+dt)
        α_tdt_test=ap_tdt[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # α_(t+dt)
        σ_tdt_test=sg_tdt[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # σ_(t+dt)
        D_tdt_test=D_tdt[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # F_(t+dt)
        
        if self.silent==False: print("... Normalizing data")
        norm_st = True
        ε_t_α, ε_t_β = self.get_α_β(ε_t_tv, norm_st)
        ε_tdt_α, ε_tdt_β = self.get_α_β(ε_t_tv+Dε_t_tv, norm_st)
        Dε_t_α, Dε_t_β = self.get_α_β(Dε_t_tv, norm_st)
        σ_t_α, σ_t_β = self.get_α_β(σ_t_tv, norm_st)
        α_t_α, α_t_β = self.get_α_β(α_t_tv, norm_st)
        F_tdt_α, F_tdt_β = self.get_α_β(F_tdt_tv, norm_01=True)
        D_tdt_α, D_tdt_β = self.get_α_β(D_tdt_tv, norm_01=True)
        σ_tdt_α, σ_tdt_β = self.get_α_β(σ_tdt_tv, norm_st)
        Dσ_tdt_α, Dσ_tdt_β = self.get_α_β(Dσ_tdt_tv, norm_st)
        α_tdt_α, α_tdt_β = self.get_α_β(α_tdt_tv, norm_st)
        Dα_tdt_α, Dα_tdt_β = self.get_α_β(Dα_tdt_tv, norm_st)
        
        n_ε_t_tv = self.out_stnd_nrml(ε_t_tv, ε_t_α, ε_t_β)
        n_Dε_t_tv = self.out_stnd_nrml(Dε_t_tv, Dε_t_α, Dε_t_β)
        n_σ_t_tv = self.out_stnd_nrml(σ_t_tv, σ_t_α, σ_t_β)
        n_α_t_tv = self.out_stnd_nrml(α_t_tv, α_t_α, α_t_β)
        n_ε_t_test = self.out_stnd_nrml(ε_t_test, ε_t_α, ε_t_β)
        n_Dε_t_test = self.out_stnd_nrml(Dε_t_test, Dε_t_α, Dε_t_β)
        n_σ_t_test = self.out_stnd_nrml(σ_t_test, σ_t_α, σ_t_β)
        n_α_t_test = self.out_stnd_nrml(α_t_test, α_t_α, α_t_β)
        n_F_tdt_tv = self.out_stnd_nrml(F_tdt_tv, F_tdt_α, F_tdt_β)
        n_D_tdt_tv = self.out_stnd_nrml(D_tdt_tv, D_tdt_α, D_tdt_β)
        n_Dσ_tdt_tv = self.out_stnd_nrml(Dσ_tdt_tv, Dσ_tdt_α, Dσ_tdt_β)
        n_Dα_tdt_tv = self.out_stnd_nrml(Dα_tdt_tv, Dα_tdt_α, Dα_tdt_β)
        n_F_tdt_test = self.out_stnd_nrml(F_tdt_test, F_tdt_α, F_tdt_β)
        n_D_tdt_test = self.out_stnd_nrml(D_tdt_test, D_tdt_α, D_tdt_β)
        n_Dσ_tdt_test = self.out_stnd_nrml(Dσ_tdt_test, Dσ_tdt_α, Dσ_tdt_β)
        n_Dα_tdt_test = self.out_stnd_nrml(Dα_tdt_test, Dα_tdt_α, Dα_tdt_β)
        
        nrm_inp=[ε_t_α, ε_t_β, ε_tdt_α, ε_tdt_β, Dε_t_α, Dε_t_β, σ_t_α, σ_t_β, α_t_α, α_t_β]
        nrm_out=[α_tdt_α, α_tdt_β, Dα_tdt_α, Dα_tdt_β, F_tdt_α, F_tdt_β, σ_tdt_α, σ_tdt_β, Dσ_tdt_α, Dσ_tdt_β, D_tdt_α, D_tdt_β]
        inputs=[n_ε_t_tv,n_Dε_t_tv,n_σ_t_tv,n_α_t_tv,n_ε_t_test,n_Dε_t_test,n_σ_t_test,n_α_t_test]
        outputs=[n_F_tdt_tv,n_D_tdt_tv,n_Dσ_tdt_tv,n_Dα_tdt_tv,n_F_tdt_test,n_D_tdt_test,n_Dσ_tdt_test,n_Dα_tdt_test]
        return nrm_inp,nrm_out,inputs,outputs
    
    def preInandOut(self,inputs,outputs):
        '''Assemble data for training, validating, and testing'''
        if self.silent==False: print("... Assembling data for training, validating, and testing")
        n_ε_t_tv,n_Dε_t_tv,n_σ_t_tv,n_α_t_tv,n_ε_t_test,n_Dε_t_test,n_σ_t_test,n_α_t_test = inputs
        n_F_tdt_tv,n_D_tdt_tv,n_Dσ_tdt_tv,n_Dα_tdt_tv,n_F_tdt_test,n_D_tdt_test,n_Dσ_tdt_test,n_Dα_tdt_test = outputs
        uN_TV = np.concatenate((np.expand_dims(n_ε_t_tv, axis=1),
                                np.expand_dims(n_Dε_t_tv, axis=1),
                                np.expand_dims(n_σ_t_tv, axis=1),
                                np.expand_dims(n_α_t_tv, axis=1)), axis=1)
        uN_T = uN_TV[:self.n_train_samples,:]
        uN_V = uN_TV[self.n_train_samples:(self.n_train_samples + self.n_val_samples),:]
        uN_Tt = np.concatenate((np.expand_dims(n_ε_t_test, axis=1),
                                np.expand_dims(n_Dε_t_test, axis=1),
                                np.expand_dims(n_σ_t_test, axis=1),
                                np.expand_dims(n_α_t_test, axis=1)), axis=1)
        oN_T = [n_Dα_tdt_tv[:self.n_train_samples],n_F_tdt_tv[:self.n_train_samples],n_Dσ_tdt_tv[:self.n_train_samples],n_D_tdt_tv[:self.n_train_samples]]
        oN_V = [n_Dα_tdt_tv[self.n_train_samples:(self.n_train_samples + self.n_val_samples)],n_F_tdt_tv[self.n_train_samples:(self.n_train_samples + self.n_val_samples)],n_Dσ_tdt_tv[self.n_train_samples:(self.n_train_samples + self.n_val_samples)],n_D_tdt_tv[self.n_train_samples:(self.n_train_samples + self.n_val_samples)]]
        oN_Tt = [n_Dα_tdt_test,n_F_tdt_test,n_Dσ_tdt_test,n_D_tdt_test]
        return uN_T,uN_V,uN_Tt,oN_T,oN_V,oN_Tt
    
    def Out(self):
        '''output data for training, validating, and testing'''
        return self.uN_T,self.uN_V,self.uN_Tt,self.oN_T,self.oN_V,self.oN_Tt,self.nrm_inp,self.nrm_out
        
    def get_α_β(self,u,norm=True,norm_01=False,no=False):
        ''' compute normalization parameters'''
        if no == False:
            if norm == True:
                if norm_01 == False:
                    u_max = np.amax(u)
                    u_min = np.amin(u)
                    u_α = (u_max - u_min) / 2.
                    u_β = (u_max + u_min) / 2.
                else:
                    u_α = np.amax(u)
                    u_β = 0.
    
            elif norm == False:
                u_α = np.std(u, axis=0)
                u_β = np.mean(u, axis=0)
        
        else:
            u_α = 1.
            u_β = 0.          
        return np.float32(u_α), np.float32(u_β)
    
    def out_stnd_nrml(self,u,α,β):
        ''' Standardize/normalize '''
        return (u-β)/α
    
    def u_stnd_nrml(self,output,α,β):
        ''' Un-standardize/un-normalize '''
        return output*α+β
