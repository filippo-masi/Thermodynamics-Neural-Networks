'''
Created on 26 Jan 2021

@author: filippomasi & ioannisstefanou
'''

from classThermodynamicsANNs.ThermodynamicsANNs import TANNs
from classThermodynamicsANNs.PreProcessingOperations import preProcessing
silent=False; silent_summary=True #silent=True: avoid info messages

# Pre-processing data
data=preProcessing('reference_data/input_data','reference_data/output_data',silent)
uN_T,uN_V,uN_Tt,oN_T,oN_V,oN_Tt,nrm_inp,nrm_out=data.Out()

# Initialize class Thermodynamics-based Artificial Neural Networks
ThermoANN=TANNs(nrm_inp,nrm_out,silent); inputs=(None,4); ThermoANN.build(inputs)
if silent_summary==False: print(ThermoANN.summary())

# Training, evaluation against training, validation, and test data-sets, and weights export
if silent==False: print("\n... Training")
learningRate=1e-4; nEpochs=10; bSize=10
historyTraining=ThermoANN.setTraining(ThermoANN,uN_T,oN_T,uN_V,oN_V,learningRate,nEpochs,bSize)
ThermoANN.evaluate(uN_T,oN_T); ThermoANN.evaluate(uN_V,oN_V); ThermoANN.evaluate(uN_Tt,oN_Tt)
if silent==False: print("\n... Saving weights")
ThermoANN.save_weights('./output_data/ThermoTANN_weights_try', save_format='tf')
print("\n... Completed!")