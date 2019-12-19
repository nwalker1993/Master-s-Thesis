##########################################################################################################
##################ANN script for the ipsilateral or contralateral leg pair mapping########################
##########################################################################################################
#Last update: 10.09.19
##########################################################################################################
#Part of the Master Thesis: Transformation of Spatial Contact Information among Limbs 
#by Nicole Walker, 2019
#This script defines and executes the neural network that maps joint angles of one leg to joint angles
#of another leg. Only pairs of legs can be mapped - either ipsilateral or contralateral.
########################################################################
#edit Nicole Walker
###########################################################################################################
###########################################################################################################
#Requirements: used dataset with angles/targets mapping (e.g. Overlap_Targets_and_JointAngles_05_tar.mat)
#					the data needs to be oranganized as follows: (samples x 6), the first three columns
#					being the sender joint angles and the latter three the receiver joint angles
###########################################################################################################
###########################################################################################################

###############################################################
############Importing packages and functions###################
###############################################################
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import json
from keras.models import model_from_json, load_model
from keras.callbacks import ModelCheckpoint, Callback
import pickle
import scipy.io
from sklearn.metrics import mean_squared_error

######################################
# Parameters ######################### 
######################################
batch_size = 10
epochs = 5000

# Number of repetitions for each architecture
run_training = 5

# Different architectures
hidden_size = [0,1,2,4,8,16,32,64,128]
#hidden_size = [0]
layers = 1

# Store the data in list
sizes = '_'.join(str(e) for e in hidden_size) #for automatizing the saving process
hist_list = []
calc_hist_list = []

#Define your leg pair mapping and the direction of mapping (anterior to posterior OR left to right -> 'forw')
#else: 'backw'
leg_pair_data = 'ANGLES_L1R1'
direction = 'forw'

######################################
########For file name saving##########
######################################
if direction == 'forw':
	leg_pair = leg_pair_data
elif direction == 'backw':
	leg_pair = 'ANGLES_' + str (leg_pair_data[-2:]) + str(leg_pair_data[-4:-2])
if leg_pair[-3] > leg_pair[-1]:
	direction = 'backw'
elif leg_pair[-3] < leg_pair[-1]:
	direction = 'forw'
else:
	direction = 'contralat'    

######################################
# Load data for training #############
######################################
mat = scipy.io.loadmat('../Synthetic Data/Overlap_Targets_And_JointAngles_05_tar.mat')
datatype = '_05_tar_div180' #for saving file name: Version of data file, tarsus included, data divided by 180

# FIRST LEG AS AN EXAMPLE: 'ANGLES_L2L3'
# TRAIN ON LEG ANGLES
# Complete data set (inputs and targets = number of samples, 6 dimensions of angles
all_data_set_load = mat[leg_pair_data]

#######################################
#######Data preprocessing##############
#######################################
# Remove data points with no real angular values (NaN) - these stem from 
# problem in inverse kinematic calculation
rm_row = []
for i in range(0,all_data_set_load.shape[0]):
    if (np.sum(np.isnan(all_data_set_load[i])) > 0):
        rm_row.append(i)
all_data_set = np.delete(all_data_set_load, rm_row, axis=0)
print(all_data_set_load.shape, all_data_set.shape)

#Divide data by 180 to fit the sigmoidal activation function
all_data_set /= 180

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
model = None
print(X_train.shape, Targets_train.shape, X_test.shape, Targets_test.shape)

######################################
########## TRAIN NETWORKS ############
######################################
######################################
# Vary Size of Hidden Layer ##########
######################################
for hidd_size in hidden_size:
	print(" ######## HIDDEN MODEL ######## ")
	print(" ######## ", hidd_size)
	print(" ######## HIDDEN MODEL ######## ")
	hist_list.append([])
	calc_hist_list.append([])
	
	# Run multiple runs for each architecture, size of hidden units
	for run_it in range(0, run_training):
		print(" ######## Trainings run ######## ")
		print(" ######## ", run_it)
		print(" ######## HIDDEN MODEL  ######## ")
		print(" ######## ", hidd_size)
		model = Sequential()
		
		# Adding the hidden layer: fully connected and using sigmoid shaped activation
		# For deeper networks you might switch towards other units if you want.
		if (hidd_size > 0):
			model.add(Dense(hidd_size, activation='sigmoid', input_dim=3))# input_shape=(3,)))
			model.add(Dense(3, activation='linear'))
			
		# When there is no hidden layer, setup simple linear model
		else:
			print("Model does not include hidden layer")
			model.add(Dense(3, activation='linear', input_dim=3))
	
		# Use MSE and Adam as an optimizer
		model.compile(loss='mean_squared_error', 
				  optimizer=Adam())
		
		#For the dictionary
		losses = ['loss', 'val_loss']
		
		# Manually calculate the loss after each epoch transfered back into angles
		class prediction_history(Callback):
			def __init__(self):
				self.predhis =  []
				self.predhis_val =  []
				
			def on_epoch_end(self, epoch, logs={}):
				pred_after_epoch_train = model.predict(X_train)
				out = pred_after_epoch_train*180
				mse = mean_squared_error(Targets_train*180, out)
				self.predhis.append(mse)
				
				pred_after_epoch_test = model.predict(X_test)
				out_val = pred_after_epoch_test*180
				mse_val = mean_squared_error(Targets_test*180, out_val)
				self.predhis_val.append(mse_val)
				
		predictions = prediction_history()

		# Start training
		history = model.fit(X_train, Targets_train,
						 batch_size=batch_size,
						 epochs=epochs,
						 verbose=0,
						 validation_data=(X_test, Targets_test), callbacks=[predictions]) 
		hist_list[-1].append(history.history) 
		
		# Compress the losses into a dictionary
		pred = [predictions.predhis,  predictions.predhis_val]
		dictionary = dict(zip(losses, pred))
		print(dictionary)
		calc_hist_list[-1].append(dictionary)
	
	# Save model and weights after each run
	model_json = model.to_json()
	with open('../Contralateral Analysis/Div180/json_'+str(run_training)+'runs_'+str(epochs)+'ep_'+str(leg_pair[7:])+'_'+str(direction)+'_'+'size_'+str(hidd_size)+'_layers'+str(layers)+str(datatype), 'w') as json_file:
		json_file.write(model_json)
	model.save_weights('../Contralateral Analysis/Div180//json_'+str(run_training)+'runs_'+str(epochs)+'ep_'+str(leg_pair[7:])+'_'+str(direction)+'_'+'size_'+str(hidd_size)+'_layers'+str(layers)+str(datatype)+'.h5')
	
#########################################################################
#####Save all network architectures and losses in one history file#######
#########################################################################	 
# Structure of the Training data - different levels:
# 1) Top level list = for different architectures (size of hidden layer):
#     [0,1,2,4,8,16,32,64,128]
# 2) Next level list: multiple training runs from random initializations, n=10
# 3) Dict: contains 'loss', 'val_loss' as keys
# 4) and as entries on next level the associated time series (2000 learning iterations)
# Loading the training data from the pickle file

# Calculated history by the network
with open('../Contralateral Analysis/Div180//trainHistoryDict_'+str(run_training)+'runs_'+str(epochs)+'ep_'+str(leg_pair[7:])+'_'+str(direction)+'_'+'size_'+str(sizes)+'_layers'+str(layers)+str(datatype), 'wb') as file_pi:
	pickle.dump(hist_list, file_pi)  

# Manually calculated loss converted back to angles (because we divided by 180)
with open('../Contralateral Analysis/Div180//calc_trainHistoryDict_'+str(run_training)+'runs_'+str(epochs)+'ep_'+str(leg_pair[7:])+'_'+str(direction)+'_'+'size_'+str(sizes)+'_layers'+str(layers)+str(datatype), 'wb') as file_pi:
	pickle.dump(calc_hist_list, file_pi) 

