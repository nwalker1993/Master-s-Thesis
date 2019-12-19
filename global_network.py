##########################################################################################################
################ANN script for a global representation mapping including all 6 legs#######################
##########################################################################################################
#Last update: 11.09.19
##########################################################################################################
#Part of the Master Thesis: Transformation of Spatial Contact Information among Limbs 
#by Nicole Walker, 2019
#This script defines and executes the neural network that maps joint angles of one leg to joint angles
#of all five other legs including a binary information on the ability of the receiver leg to reach the
#target point.
########################################################################
#edit Nicole Walker
###########################################################################################################
###########################################################################################################
#Requirements: used dataset with angles/targets mapping (e.g. synth_dataset.mat)
#					the data needs to be organized as follows: (samples x 23), the first three columns
#					being the sender joint angles, the other 20 columns being the receiver legs with 
#					4 columns per leg for three joint angles and a column for binary information on the 
#					ability to reach the target point (If 0, the joint angles of the receiver leg are the 
#					closest in Euclidean distance to the target point that are still reachable). 
#					The receiver legs need to be ordered according to L1, L2, L3, R1, R2, R3
###########################################################################################################
###########################################################################################################

###############################################################
############Importing packages and functions###################
###############################################################
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import json
from keras.models import model_from_json, load_model
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import plot_model
import pickle
import scipy.io
from sklearn.metrics import mean_squared_error

######################################
# Parameters ######################### 
######################################
batch_size = 10
epochs = 1000

# Number of repetitions for each architecture
run_training = 5

# Different architectures
#hidden_size = [0,1,2,4,8,16,32,64,128]
hidden_size = [32]
layers = 1

# Store the data in list
sizes = '_'.join(str(e) for e in hidden_size) #for automatizing the saving process
hist_list = []
calc_hist_list = []

# Define the sender leg according to the order L1, L2, L3, R1, R2, R3 (zero indexing!)
sender = 2
legs = ['L1','L2', 'L3', 'R1', 'R2', 'R3']
leg_pair = ''.join(str(e) for e in legs[0:6])
print(leg_pair)

#For labeling the layer names and the dictionary
if sender == 0:
	legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
	losses = ['loss','L2_loss', 'L3_loss', 'R1_loss', 'R2_loss', 'R3_loss', 'val_loss', 'val_L2_loss', 'val_L3_loss', 'val_R1_loss', 'val_R2_loss', 'val_R3_loss']
elif sender == 1:
	legs = ['L2', 'L1', 'L3', 'R1', 'R2', 'R3']
	losses = ['loss','L1_loss', 'L3_loss', 'R1_loss', 'R2_loss', 'R3_loss', 'val_loss', 'val_L1_loss', 'val_L3_loss', 'val_R1_loss', 'val_R2_loss', 'val_R3_loss']
elif sender == 2:
	legs = ['L3', 'L1', 'L2', 'R1', 'R2', 'R3']
	losses = ['loss','L1_loss', 'L2_loss', 'R1_loss', 'R2_loss', 'R3_loss', 'val_loss', 'val_L1_loss', 'val_L2_loss', 'val_R1_loss', 'val_R2_loss', 'val_R3_loss']
elif sender == 3:
	legs = ['R1', 'L1', 'L2', 'L3', 'R2', 'R3']
	losses = ['loss','L1_loss', 'L2_loss', 'L3_loss', 'R2_loss', 'R3_loss', 'val_loss', 'val_L1_loss', 'val_L2_loss', 'val_L3_loss', 'val_R2_loss', 'val_R3_loss']
elif sender == 4:
	legs = ['R2', 'L1', 'L2', 'L3', 'R1', 'R3']
	losses = ['loss','L1_loss', 'L2_loss', 'L3_loss', 'R1_loss', 'R3_loss', 'val_loss', 'val_L1_loss', 'val_L2_loss', 'val_L3_loss', 'val_R1_loss', 'val_R3_loss']
elif sender == 5:
	legs = ['R3', 'L1', 'L2', 'L3', 'R1', 'R2']
	losses = ['loss','L1_loss', 'L2_loss', 'L3_loss', 'R1_loss', 'R2_loss', 'val_loss', 'val_L1_loss', 'val_L2_loss', 'val_L3_loss', 'val_R1_loss', 'val_R2_loss']

######################################
# Load data for training #############
######################################
mat = scipy.io.loadmat('../Synthetic Data/synth_dataset_L360.mat')

# TRAIN ON LEG ANGLES
load_angles = mat['ANGLES_map_yn'] 
load_angles = load_angles.squeeze()

all_data_set =  load_angles[sender]
only_angles = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21]
all_data_set[:,only_angles] /= 180

#######################################
#######Data preprocessing##############
#######################################
# Randomly draw indices for training and test set:
indices = np.random.permutation(all_data_set.shape[0])
training_idx, test_idx = indices[:int(0.8 * all_data_set.shape[0])], indices[int(0.8 * all_data_set.shape[0]):]

# Construct training and test set
training_data, test_data = all_data_set[training_idx,:], all_data_set[test_idx,:]

# Cutting training and test set into input data X and target values
#Training set
train_list = training_data
train_list = np.array_split(train_list, [3, 7, 11, 15, 19], axis=1) 
X_train = train_list[0]
Target1_train = train_list[1]
Target2_train = train_list[2]
Target3_train = train_list[3]
Target4_train = train_list[4]
Target5_train = train_list[5]

#Test set
test_list = test_data
test_list = np.array_split(test_list, [3, 7, 11, 15, 19], axis=1)
X_test = test_list[0]
Target1_test = test_list[1]
Target2_test = test_list[2]
Target3_test = test_list[3]
Target4_test = test_list[4]
Target5_test = test_list[5]
print(X_train.shape, Target1_train.shape, Target2_train.shape, X_test.shape, Target1_test.shape, Target2_test.shape)

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
		
		#Input layer
		inputs = Input(shape=(3,), name=legs[0])
		
		# Adding the hidden layer: fully connected and using sigmoid shaped activation
		# For deeper networks you might switch towards other units if you want.
		if (hidd_size > 0):
			x = Dense(hidd_size, activation='sigmoid', name='Hidden')(inputs)
		
			Target1_out = Dense(4, activation='linear', name=legs[1])(x)
			Target2_out = Dense(4, activation='linear', name=legs[2])(x)
			Target3_out = Dense(4, activation='linear', name=legs[3])(x)
			Target4_out = Dense(4, activation='linear', name=legs[4])(x)
			Target5_out = Dense(4, activation='linear', name=legs[5])(x)
			
		# When there is no hidden layer, setup simple linear model
		else:
			print("Model does not include hidden layer")
			Target1_out = Dense(4, activation='linear', name=legs[1])(inputs)
			Target2_out = Dense(4, activation='linear', name=legs[2])(inputs)
			Target3_out = Dense(4, activation='linear', name=legs[3])(inputs)
			Target4_out = Dense(4, activation='linear', name=legs[4])(inputs)
			Target5_out = Dense(4, activation='linear', name=legs[5])(inputs)
		
		# Defines the functional model
		model = Model(inputs=inputs, outputs=[Target1_out, Target2_out, Target3_out, Target4_out, Target5_out])
		
		# Use MSE and Adam as an optimizer
		model.compile(loss='mean_squared_error', 
						  optimizer=Adam())
		
		# Manually calculate the loss after each epoch transfered back into angles
		class prediction_history(Callback):
			def __init__(self):
				self.predhis_X =  []
				self.predhis_Target1 = []
				self.predhis_Target2 = []
				self.predhis_Target3 = []
				self.predhis_Target4 = []
				self.predhis_Target5 = []
				
				self.predhis_X_val =  []
				self.predhis_Target1_val = []
				self.predhis_Target2_val = []
				self.predhis_Target3_val = []
				self.predhis_Target4_val = []
				self.predhis_Target5_val = []
				
			def on_epoch_end(self, epoch, logs={}):
				#training error
				pred_after_epoch_train = model.predict(X_train)
				out_Target1 = pred_after_epoch_train[0][:,0:3]*180
				out_Target2 = pred_after_epoch_train[1][:,0:3]*180
				out_Target3 = pred_after_epoch_train[2][:,0:3]*180
				out_Target4 = pred_after_epoch_train[3][:,0:3]*180
				out_Target5 = pred_after_epoch_train[4][:,0:3]*180

				mse_Target1, mse_Target2, mse_Target3, mse_Target4, mse_Target5 = mean_squared_error(Target1_train[:,0:3]*180, out_Target1), mean_squared_error(Target2_train[:,0:3]*180, out_Target2), mean_squared_error(Target3_train[:,0:3]*180, out_Target3), mean_squared_error(Target4_train[:,0:3]*180, out_Target4), mean_squared_error(Target5_train[:,0:3]*180, out_Target5)
				
				mse = mse_Target1+mse_Target2+mse_Target3+mse_Target4+mse_Target5
				
				self.predhis_X.append(mse)
				self.predhis_Target1.append(mse_Target1)
				self.predhis_Target2.append(mse_Target2)
				self.predhis_Target3.append(mse_Target3)
				self.predhis_Target4.append(mse_Target4)
				self.predhis_Target5.append(mse_Target5)
				
				#test error
				pred_after_epoch_test = model.predict(X_test)
				out_Target1_val = pred_after_epoch_test[0][:,0:3]*180
				out_Target2_val = pred_after_epoch_test[1][:,0:3]*180
				out_Target3_val = pred_after_epoch_test[2][:,0:3]*180
				out_Target4_val = pred_after_epoch_test[3][:,0:3]*180
				out_Target5_val = pred_after_epoch_test[4][:,0:3]*180
				
				mse_Target1_val, mse_Target2_val, mse_Target3_val, mse_Target4_val, mse_Target5_val = mean_squared_error(Target1_test[:,0:3]*180, out_Target1_val), mean_squared_error(Target2_test[:,0:3]*180, out_Target2_val), mean_squared_error(Target3_test[:,0:3]*180, out_Target3_val), mean_squared_error(Target4_test[:,0:3]*180, out_Target4_val), mean_squared_error(Target5_test[:,0:3]*180, out_Target5_val)
				
				mse_val = mse_Target1_val+mse_Target2_val+mse_Target3_val+mse_Target4_val+mse_Target5_val
				
				self.predhis_X_val.append(mse_val)
				self.predhis_Target1_val.append(mse_Target1_val)
				self.predhis_Target2_val.append(mse_Target2_val)
				self.predhis_Target3_val.append(mse_Target3_val)
				self.predhis_Target4_val.append(mse_Target4_val)
				self.predhis_Target5_val.append(mse_Target5_val)
				
		predictions = prediction_history()
			
		# Start training
		history = model.fit(X_train, [Target1_train, Target2_train, Target3_train, Target4_train, Target5_train],
								 batch_size=batch_size,
								 epochs=epochs,
								 verbose=0,
								 validation_data=(X_test, [Target1_test, Target2_test, Target3_test, Target4_test, Target5_test]),
								 callbacks=[predictions]) 	
		hist_list[-1].append(history.history) 
		
		# Compress the losses into a dictionary
		pred = [predictions.predhis_X, predictions.predhis_Target1, predictions.predhis_Target2, predictions.predhis_Target3, predictions.predhis_Target4, predictions.predhis_Target5,
					predictions.predhis_X_val, predictions.predhis_Target1_val, predictions.predhis_Target2_val, predictions.predhis_Target3_val, predictions.predhis_Target4_val, predictions.predhis_Target5_val]	
		dictionary = dict(zip(losses, pred))
		print(dictionary)
		calc_hist_list[-1].append(dictionary)
		
		# Save model and weights after each run
		model_json = model.to_json()
		with open('json_'+str(run_training)+'runs_'+str(epochs)+'ep_'+str(leg_pair)+'_senderleg_'+str(legs[0])+'_'+'size_'+str(hidd_size)+'_run_'+str(run_it+1)+'global_180tar_4outbin_0toEuc_60', 'w') as json_file:
			json_file.write(model_json)
		model.save_weights('json_'+str(run_training)+'runs_'+str(epochs)+'ep_'+str(leg_pair)+'_senderleg_'+str(legs[0])+'_'+'size_'+str(hidd_size)+'_run_'+str(run_it+1)+'global_180tar_4outbin_0toEuc_60.h5')
    

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
with open('trainHistoryDict_'+str(run_training)+'runs_'+str(epochs)+'ep_'+str(leg_pair)+'_senderleg_'+str(legs[0])+'_'+'size_'+str(sizes)+'global_180tar_4outbin_0toEuc_60', 'wb') as file_pi:
    pickle.dump(hist_list, file_pi)    

# Manually calculated loss converted back to angles (because we divided by 180)
with open('calc_trainHistoryDict_'+str(run_training)+'runs_'+str(epochs)+'ep_'+str(leg_pair)+'_senderleg_'+str(legs[0])+'_'+'size_'+str(sizes)+'global_180tar_4outbin_0toEuc_60', 'wb') as file_pi:
    pickle.dump(calc_hist_list, file_pi) 
