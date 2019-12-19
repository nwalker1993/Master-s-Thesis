##########################################################################################################
###############Script to produce individual MSE for each target point and joint angle#####################
##########################################################################################################
#Last update: 09.09.19
##########################################################################################################
#Part of the Master Thesis: Transformation of Spatial Contact Information among Limbs 
#by Nicole Walker, 2019
#This script predicts the MSE per sample point and per joint angle and is used as an input to plotleg.py. 
#As input it takes the sender leg, the saved weights as JSON file and the sample point considered.
########################################################################
#edit Nicole Walker
###########################################################################################################
###########################################################################################################
#Requirements: used dataset with angles/targets mapping (e.g. synth_dataset_02.mat)
#			  
###########################################################################################################
###########################################################################################################

###############################################################
############Importing packages and functions###################
###############################################################
import numpy as np
import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib.pylab as py
import pickle
import scipy.io

##############################################################
####################Define the function#######################
##############################################################
def error_map(sender, model_file, N):
	#################################################################
	##########################Parameters#############################
	#################################################################
	name_in = model_file.index('json')
	name = model_file[name_in+len('json_'):]
	
	if sender == 'L1':
		legs = ['L2', 'L3', 'R1', 'R2', 'R3']
		sender_leg = 0
	elif sender == 'L2':
		legs = ['L1', 'L3', 'R1', 'R2', 'R3']
		sender_leg = 1
	elif sender == 'L3':
		legs = ['L1', 'L2', 'R1', 'R2', 'R3']
		sender_leg = 2
	elif sender == 'R1':
		legs = ['L1', 'L2', 'L3', 'R2', 'R3']
		sender_leg = 3
	elif sender == 'R2':
		legs = ['L1', 'L2', 'L3', 'R1', 'R3']
		sender_leg = 4
	elif sender == 'R3':
		legs = ['L1', 'L2', 'L3', 'R1', 'R2']
		sender_leg = 5
	range1 = np.arange(3,6)
	range2 = np.arange(7,10)
	range3 = np.arange(11,14)
	range4 = np.arange(15,18)
	range5 = np.arange(19,22)
			
	#score_leg = 0
	#leg = np.arange(3,6)
	#################################################################
	#############Data loading and preprocessing######################
	#################################################################
	mat = scipy.io.loadmat('../Synthetic Data/synth_dataset_02.mat')
	all_angles_set_load = mat['ANGLES_map_yn']
	all_target_set_load = mat['TARGET_map_clean']
	all_target_set = all_target_set_load[0,sender_leg]
	all_angles_set = all_angles_set_load[0,sender_leg]/180
	
	'''mat = scipy.io.loadmat('../Synthetic Data/multiple_test_02_tar.mat')
	all_angles_set = mat['ANGLES_R1R2R3']
	all_target_set = mat['TARGET_R1R2R3']
	all_angles_set = all_angles_set/180'''
	
	forw_angles_set, forw_target_set = all_angles_set[N,0:3], all_target_set[N,0:3]
	print(all_angles_set_load.shape, all_angles_set.shape, all_target_set_load.shape, all_target_set.shape)
	print(forw_angles_set.reshape((1,3)).shape)
	#################################################################
	#######Load the trained network you want to use##################
	#################################################################
	###Loads the architecture###
	json_file = open(model_file, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	###Loads the weights###
	loaded_model.load_weights(model_file + '.h5')

	#################################################################
	########Manually compute the MSE for each target point###########
	#################################################################
	class errors():
		def __init__(self, leg_range, rec):
			self.leg_range = leg_range
			self.rec = rec
			#predict the dataset with the trained network 
			self.score = loaded_model.predict(forw_angles_set.reshape((1,3)), verbose = 0)
			self.angles = self.score[self.rec][0][0:3]*180
			#calculate the difference between true values and prediction
			self.df = np.copy(all_angles_set[N,self.leg_range]*180) - np.copy(self.score[self.rec][0][0:3]*180)

			#calculate the mean of the squared difference
			self.mse = np.mean(np.square(np.copy(self.df.reshape((1,3)))), axis = 1)
			self.mse_pj = np.mean(np.square(np.copy(self.df.reshape((1,3)))), axis = 0)
			print('############# '+str(legs[self.rec])+' #############')
			print('Input angles: '+str(forw_angles_set*180))
			print('Output angles: '+str(all_angles_set[N,self.leg_range]*180))
			print('Predicted Angles: '+str(self.angles))
			print('Difference between respective angles: '+str('alpha: {0:.3f}'.format(self.df[0]))+str(', beta: {0:.3f}'.format(self.df[1]))+str(', gamma: {0:.3f}'.format(self.df[2])))
			print('MSE per angle: ' +str('alpha: {0:.3f}'.format(self.mse_pj[0]))+str(', beta: {0:.3f}'.format(self.mse_pj[1]))+str(', gamma: {0:.3f}'.format(self.mse_pj[2])))
			print('MSE total: ' +str('{0:.3f}'.format(self.mse[0])))
			print('Mean difference per angle: '+str('{0:.3f}'.format(np.sqrt(self.mse)[0])))
	
	er_rec1 = errors(range1,0)	
	er_rec2 = errors(range2,1)
	er_rec3 = errors(range3,2)
	er_rec4 = errors(range4,3)
	er_rec5 = errors(range5,4)
	
	return er_rec1, er_rec2, er_rec3, er_rec4, er_rec5

