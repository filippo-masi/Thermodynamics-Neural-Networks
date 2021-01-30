'''
Created on 26 Jan 2021

@author: filippomasi & ioannisstefanou
'''

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

def act(x):
    return tf.keras.backend.elu(tf.keras.backend.pow(x, 2), alpha=1.0) 
tf.keras.utils.get_custom_objects().update({'act': tf.keras.layers.Activation(act)})

class TANNs(tf.keras.Model):
    def __init__(self,nrm_inp,nrm_out,training_silent=True):
        self.training_silent=training_silent
        super(TANNs, self).__init__()
        self.f0=tf.keras.layers.Dense(6, activation=tf.nn.leaky_relu)
        self.f1=tf.keras.layers.Dense(1, use_bias=False)
        self.f2=tf.keras.layers.Dense(9, activation='act')
        self.f3=tf.keras.layers.Dense(1, use_bias=False)
        self.ε_t_α,self.ε_t_β,self.ε_tdt_α,self.ε_tdt_β,self.Dε_t_α,self.Dε_t_β,self.σ_t_α,self.σ_t_β,self.α_t_α,self.α_t_β=nrm_inp
        self.α_tdt_α,self.α_tdt_β,self.Dα_tdt_α,self.Dα_tdt_β,self.F_tdt_α,self.F_tdt_β,self.σ_tdt_α,self.σ_tdt_β,self.Dσ_tdt_α,self.Dσ_tdt_β,self.D_tdt_α,self.D_tdt_β=nrm_out
        
    def tf_out_stnd_nrml(self,u,α,β):
        ''' Standardize/normalize '''
        return tf.divide(tf.add(u,-β),α)
    
    def tf_u_stnd_nrml(self,output,α,β):
        ''' Un-standardize/un-normalize '''
        return tf.add(tf.multiply(output,α),β)
    
    def tf_get_gradients(self,f_tdt,u_ε_tdt,u_α_tdt,u_Dα_tdt,u_σ_t):
        ''' Compute normalized tf.gradients from normalized f @ t+dt and un-normalized ε and α'''
        f_σ_tdt=tf.gradients(f_tdt,u_ε_tdt)[0]  # σ @ t+dt (non-normalized)
        f_Dσ_tdt=tf.math.subtract(f_σ_tdt,u_σ_t)  # Dσ @ t+dt (non-normalized)
        nf_Dσ_tdt=self.tf_out_stnd_nrml(f_Dσ_tdt,self.Dσ_tdt_α,self.Dσ_tdt_β) # σ @ t+dt (normalized)
        f_chi_tdt=-tf.gradients(f_tdt,u_α_tdt)[0]
        f_d_tdt=tf.math.multiply(f_chi_tdt,u_Dα_tdt)
        nf_d_tdt=self.tf_out_stnd_nrml(f_d_tdt,self.D_tdt_α,self.D_tdt_β)
        return nf_Dσ_tdt,nf_d_tdt
    

    def call(self,un):   
        # Slice ε # t+dt from inputs of ANN no.1
        un_ε_t_f = tf.slice(un,[0,0],[-1,1])
        un_Dε_t_f = tf.slice(un,[0,1],[-1,1])
        un_σ_t_f = tf.slice(un,[0,2],[-1,1])
        un_α_t_f = tf.slice(un,[0,3],[-1,1])
        # Un-normalized ε_tdt, α_t, and α_tdt
        u_ε_t = self.tf_u_stnd_nrml(un_ε_t_f, self.ε_t_α, self.ε_t_β)
        u_Dε_t = self.tf_u_stnd_nrml(un_Dε_t_f, self.Dε_t_α, self.Dε_t_β)
        u_ε_tdt = tf.math.add(u_ε_t,u_Dε_t)
        u_α_t = self.tf_u_stnd_nrml(un_α_t_f, self.α_t_α, self.α_t_β)
        u_σ_t = self.tf_u_stnd_nrml(un_σ_t_f, self.σ_t_α, self.σ_t_β)
        # Re-normalized ε_tdt, α_t, and α_tdt
        nu_ε_tdt = self.tf_out_stnd_nrml(u_ε_tdt, self.ε_tdt_α, self.ε_tdt_β)
        nu_Dε_t = self.tf_out_stnd_nrml(u_Dε_t, self.Dε_t_α, self.Dε_t_β)
        nu_α_t = self.tf_out_stnd_nrml(u_α_t, self.α_t_α, self.α_t_β)
        nu_σ_t = self.tf_out_stnd_nrml(u_σ_t, self.σ_t_α, self.σ_t_β)
        un_concat_sub = tf.concat([nu_ε_tdt, nu_Dε_t, nu_α_t, nu_σ_t], 1, name = 'x1')
        # ANN no.1
        nf0_sub = self.f0(un_concat_sub)
        nf_Dα_tdt = self.f1(nf0_sub) #α_tdt (normalized)
        u_Dα_tdt = self.tf_u_stnd_nrml(nf_Dα_tdt, self.Dα_tdt_α, self.Dα_tdt_β)
        u_α_tdt = tf.math.add(u_Dα_tdt, u_α_t)
        nu_α_tdt = self.tf_out_stnd_nrml(u_α_tdt, self.α_tdt_α, self.α_tdt_β)
        # Concatenate inputs (re-)normalized
        nu_concat = tf.concat([nu_ε_tdt, nu_α_tdt], 1, name = 'a2')
        # ANN no.2
        nf1 = self.f2(nu_concat)
        nf_tdt = self.f3(nf1) # energy @ t+dt (normalized)
        f_tdt = self.tf_u_stnd_nrml(nf_tdt, self.F_tdt_α, self.F_tdt_β) # energy @ t+dt (non-normalized)
        nf_Dσ_tdt, nf_d_tdt = self.tf_get_gradients(f_tdt, u_ε_tdt, u_α_tdt, u_Dα_tdt, u_σ_t)
        return nf_Dα_tdt, nf_tdt, nf_Dσ_tdt, nf_d_tdt
    
    def setTraining(self,TANN,input_training,output_training,input_validation,output_validation,learningRate,nEpochs,bSize):
        if self.training_silent==False: silent_verbose=2
        else: silent_verbose=0
        
        optimizer = tf.keras.optimizers.Nadam(learningRate)
        tol_e=1.e+1; tol_s=1.e-1; wα=tol_e;wF=tol_s;wσ=2*tol_s;wD=tol_e*tol_s 
        TANN.compile(optimizer=optimizer,loss=['mae','mae', 'mae', 'mae'],loss_weights=[wα,wF,wσ,wD])
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=1.e-6,
                                                              patience=2000,verbose=0,
                                                              mode='auto',baseline=None,
                                                              restore_best_weights=True)
        history = TANN.fit(input_training,output_training,
                           epochs=nEpochs,batch_size=bSize,
                           verbose=silent_verbose,
                           validation_data=(input_validation,output_validation),
                           callbacks=[earlystop_callback])
        if self.training_silent==False: print("\n Training completed in", nEpochs, " epochs")
        return history