##########################################################################################################
#####Script to calculate simple regression on a given data set to compate with hidden layer size = 0 #####
##########################################################################################################
#Last update: 11.09.19
##########################################################################################################
#Part of the Master Thesis: Transformation of Spatial Contact Information among Limbs 
#by Nicole Walker, 2019
#This script calculates the predicted mean squared error from simple regression given a certain data set.
#It is implementeed as a function and used for plotting to compare to neural networks with hidden layer
#size = 0. The function can be used for individual leg pair mappings. The function
#takes the argument legpair which is the mapping used (e.g. R1R2) The other argument it takes is direction, 
#it can either be 'forw' (anterior to posterior or left to right for contralateral mappings) or 'backw' 
#(posterior to anterior or right to left for contralateral mappings). The last argument is the data set used.
########################################################################
#edit Nicole Walker
###########################################################################################################
###########################################################################################################
#Requirements: used dataset with angles/targets mapping (e.g. Overlap_Targets_and_JointAngles_05_tar.mat)
#
###########################################################################################################
###########################################################################################################

###############################################################
############Importing packages and functions###################
###############################################################
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import keras
import scipy.io

def regr(legpair, direction, dataset):
	######################################
	###Regression of given data set#######
	######################################
	# For either ipsilateral or contralateral mappings
	# Load data
	mat = scipy.io.loadmat('../Synthetic Data/'+str(dataset))
	
	# FIRST LEG AS AN EXAMPLE: 'ANGLES_L2L3'
	# TRAIN ON LEG ANGLES
	# Complete data set (inputs and targets = number of samples, 6 dimensions of angles
	all_data_set_load = mat['ANGLES_'+str(legpair)]
	
	# Remove data points with no real angular values (=1.0e+32)/NaN - these stem from 
	# problem in inverse kinematic calculation
	rm_row = []
	for i in range(0,all_data_set_load.shape[0]):
		if (np.sum(np.isnan(all_data_set_load[i])) > 0):
			rm_row.append(i)
	all_data_set = np.delete(all_data_set_load, rm_row, axis=0)
	rm_row = []
	for i in range(0,all_data_set.shape[0]):
		if (np.sum(all_data_set[i]) > 1.0e+32):
			rm_row.append(i)
	all_data_set = np.delete(all_data_set, rm_row, axis=0)
	print(all_data_set_load.shape, all_data_set.shape)

	# Randomly draw indices for training and test set:
	indices = np.random.permutation(all_data_set.shape[0])
	training_idx, test_idx = indices[:int(0.8 * all_data_set.shape[0])], indices[int(0.8 * all_data_set.shape[0]):]
	
	# Construct training and test set
	training_data, test_data = all_data_set[training_idx,:], all_data_set[test_idx,:]
	
	# Cutting training and test set into input data X and target values
	train_list = np.hsplit(training_data,2)
	test_list = np.hsplit(test_data,2)
	if direction == 'forw':
		X_train = train_list[0]
		Targets_train = train_list[1]
		X_test = test_list[0]
		Targets_test = test_list[1]
	elif direction == 'backw':
		X_train = train_list[1]
		Targets_train = train_list[0]
		X_test = test_list[1]
		Targets_test = test_list[0]
	
	# Linear regression
	regr = linear_model.LinearRegression()
	regr.fit(X_train, Targets_train)
	prediction = regr.predict(X_test)
	print('Coefficients: \n', regr.coef_)
	print('Mean squared error: %.2f', mean_squared_error(Targets_test, prediction))
	print('Variance score: %.2f' % r2_score(Targets_test, prediction)) #1 means perfect prediction
	
	return mean_squared_error(Targets_test, prediction)

	
