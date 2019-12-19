##########################################################################################################
##############Script to visualize the loss of a given neural network including all six legs###############
##########################################################################################################
#Last update: 19.09.19
##########################################################################################################
#Part of the Master Thesis: Transformation of Spatial Contact Information among Limbs 
#by Nicole Walker, 2019
#This script visualizes the training history loss of a given network for a global mapping to all 5 output
#legs. It compares the loss of over different architectures and over epochs for a single architecture.
########################################################################
#edit Nicole Walker
###########################################################################################################
###########################################################################################################
#Requirements: training histories of the neural networks 
#					saved in the form: trainHistoryDict_5runs_2500ep_L1L2L3R1R2R3_senderleg_L1
###########################################################################################################
###########################################################################################################

###############################################################
############Importing packages and functions###################
###############################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from Tkinter import Tk
import Tkinter, Tkconstants, tkFileDialog
import pickle
import scipy.io
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, matthews_corrcoef

###########################################
# Loading the training data and structure #
###########################################
# Structure of the Training data - different levels:
# 1) Top level list = for different architectures (size of hidden layer):
#     [0,1,2,4,8,16,32,64,128]
# 2) Next level list: multiple training runs from random initializations, n=10
# 3) Dict: contains 'loss', 'val_loss' as keys
# 4) and as entries on next level the associated time series (2000 learning iterations)
# Loading the training data from the pickle file

root = Tk()
root.filename = tkFileDialog.askopenfilename()
print(root.filename)
with open(root.filename, 'rb') as file_pi:
    hist_list = pickle.load(file_pi) 

name_in = root.filename.index('Dict_')
name = root.filename[name_in+len('Dict_'):]
root.destroy()

#For automatizing the plot labeling
if 'senderleg_L1' in root.filename:
	mapp = 'Sender leg L1'
	legs = ['L2', 'L3', 'R1', 'R2', 'R3']
	sender = 0
elif 'senderleg_L2' in root.filename:
	mapp = 'Sender leg L2'
	legs = ['L1', 'L3', 'R1', 'R2', 'R3']
	sender = 1
elif 'senderleg_L3' in root.filename:
	mapp = 'Sender leg L3'
	legs = ['L1', 'L2', 'R1', 'R2', 'R3']
	sender = 2
elif 'senderleg_R1' in root.filename:
	mapp = 'Sender leg R1'
	legs = ['L1', 'L2', 'L3', 'R2', 'R3']
	sender = 3
elif 'senderleg_R2' in root.filename:
	mapp = 'Sender leg R2'
	legs = ['L1', 'L2', 'L3', 'R1', 'R3']
	sender = 4
elif 'senderleg_R3' in root.filename:
	mapp = 'Sender leg R3'
	legs = ['L1', 'L2', 'L3', 'R1', 'R2']
	sender = 5
	
# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 

##########################################
# 1 A - Construct Data for 3D surface plot #
##########################################
class plotting_vals():
	def __init__(self, loss):
		self.loss = loss
		# Construct target arrays for the 3D surface visualisation data
		# Single time series (up to 2000)
		vis_epochs = np.arange(0, len(hist_list[0][0][loss]))
		# Number of variations of architectures
		vis_hid_var = np.arange(0,len(hist_list))
		# Construct meshgrid of the two different arrays
		# = three 2 dimensional arrays, vis_X and vis_Y define regular locations in two dimensional space
		#   while vis_val_loss provides the specific values
		self.vis_X, self.vis_Y = np.meshgrid(vis_epochs, vis_hid_var)
		self.vis_val_loss = np.zeros(self.vis_X.shape)

		# Pushing the loaded data into this grid structure (vis_val_loss)
		# For this: the mean over the multiple runs for a single architecture is first calculated
		for arch_n in range(0,len(hist_list)):
			copied_val_loss = np.zeros((len(hist_list[arch_n]), len(hist_list[arch_n][0][loss]) ))
			for diff_runs in range(0, len(hist_list[arch_n])):
				copied_val_loss[diff_runs] = hist_list[arch_n][diff_runs][loss]
			self.vis_val_loss[arch_n] = np.log(np.mean(copied_val_loss, axis=0)) # np.log(np.array(hist_list[i]['val_loss']))
			#vis_val_loss[i][vis_val_loss[i]>20]= 20
			
		# Construct data at end of training
		copied_arch_val_loss = np.zeros((len(hist_list), len(hist_list[0])))
		for arch_n in range(0,len(hist_list)):
			for diff_runs in range(0, len(hist_list[arch_n])):
				# Getting the last loss - at end of training
				copied_arch_val_loss[arch_n][diff_runs] = hist_list[arch_n][diff_runs][loss][-1]
		self.mean_arch_val_loss = np.mean(copied_arch_val_loss, axis=1)
		#print(copied_arch_val_loss[2])
		self.std_arch_val_loss = np.std(copied_arch_val_loss, axis=1)
		self.arch_val_loss_lower_std = self.mean_arch_val_loss - self.std_arch_val_loss
		self.arch_val_loss_upper_std = self.mean_arch_val_loss + self.std_arch_val_loss 

receiver1 = plotting_vals('val_'+legs[0]+'_loss')
receiver2 = plotting_vals('val_'+legs[1]+'_loss')
receiver3 = plotting_vals('val_'+legs[2]+'_loss')
receiver4 = plotting_vals('val_'+legs[3]+'_loss')
receiver5 = plotting_vals('val_'+legs[4]+'_loss') 

###########################################################
# 1 C - Comparison different Architectures after Training #
###########################################################
fig = plt.figure(figsize=(10, 8))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False)  
ax_arch.set_yscale('log')
ax_arch.set_xlim(-1, 9)  
ax_arch.set_xticks(np.arange(0,len(hist_list)))
ax_arch.set_xticklabels(['No hid.','1','2','4','8','16','32','64','128'])

# Use matplotlib's fill_between() call to create error bars.
plt.fill_between(range(0,len(receiver1.mean_arch_val_loss)), receiver1.arch_val_loss_lower_std,  
                 receiver1.arch_val_loss_upper_std, color='g', alpha=0.5) 

plt.fill_between(range(0,len(receiver2.mean_arch_val_loss)), receiver2.arch_val_loss_lower_std,  
                 receiver2.arch_val_loss_upper_std, color='r', alpha=0.5) 
                 
plt.fill_between(range(0,len(receiver3.mean_arch_val_loss)), receiver3.arch_val_loss_lower_std,  
                 receiver3.arch_val_loss_upper_std, color='y', alpha=0.5) 
   
plt.fill_between(range(0,len(receiver4.mean_arch_val_loss)), receiver4.arch_val_loss_lower_std,  
                 receiver4.arch_val_loss_upper_std, color=tableau20[1], alpha=0.5) 
                 
plt.fill_between(range(0,len(receiver5.mean_arch_val_loss)), receiver5.arch_val_loss_lower_std,  
                 receiver5.arch_val_loss_upper_std, color=tableau20[3], alpha=0.5) 

plt.plot(range(0,len(receiver1.mean_arch_val_loss)), receiver1.mean_arch_val_loss, color='g', lw=2, label=legs[0]+' output')
plt.plot(range(0,len(receiver2.mean_arch_val_loss)), receiver2.mean_arch_val_loss, color='r', lw=2, label=legs[1]+' output')
plt.plot(range(0,len(receiver3.mean_arch_val_loss)), receiver3.mean_arch_val_loss, color='y', lw=2, label=legs[2]+' output')
plt.plot(range(0,len(receiver4.mean_arch_val_loss)), receiver4.mean_arch_val_loss, color=tableau20[0], lw=2, label=legs[3]+' output')
plt.plot(range(0,len(receiver5.mean_arch_val_loss)), receiver5.mean_arch_val_loss, color=tableau20[2], lw=2, label=legs[4]+' output')
plt.legend(frameon = False, fontsize=16)
plt.plot([-1,8], [10,10], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,8], [1,1], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,8], [0.1,0.1], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,8], [100,100], '-', color='gray', lw=1, alpha=0.25)
ax_arch.set_xlabel('# Hidden units', fontsize=20, labelpad=17)
ax_arch.set_ylabel('MSE', fontsize=20)
plt.tick_params(labelsize=19)  
plt.ylim(10**(0), 10**3)
ax_arch.set_title('MSE over different Architectures \n'+mapp, fontsize=25) 
plt.tick_params(labelsize=19)
fig.savefig('Figures/'+str(name)+'_MSEArch', dpi=300, bbox_inches='tight')



#####################################################
########Seperate the error for 0 and 1 trials########
#####################################################
#as many trials have few overlaps but still need to have the same output size, the ratio between "real"
#mappings and minimum Euclidean distance mappings is small. Therefore we seperate the predictions for those two

#load data and shape to needs
mat = scipy.io.loadmat('../Synthetic Data/synth_dataset_02.mat')
load_angles = mat['ANGLES_map_yn'] 
load_angles = load_angles.squeeze()
all_data_set =  load_angles[sender]
all_data_set_bin = np.copy(all_data_set)
all_data_set /= 180
only_angles = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21]
all_data_set_bin[:,only_angles] /= 180

#for loading the files in the loop and saving the values in a dictionary
hidden_size = [0,1,2,4,8,16,32,64,128]
mse_list = []
run_training = 5
receivers = ['reach_r1', 'nonreach_r1', 'reach_r2', 'nonreach_r2', 'reach_r3', 'nonreach_r3', 'reach_r4', 'nonreach_r4', 'reach_r5', 'nonreach_r5']
name_size = name.index('size_')
name_type = name.index('global')

#loops over architectures and runs, the shape of the dictionary is the same as before for trainDict
for hidd_size in range(0,len(hidden_size)):
	mse_list.append([])
	for run_it in range(0, run_training):
		#reads and loads the model architecture/weights files
		json_file = open('json_'+ name[:name_size+len('size_')]+ str(hidden_size[hidd_size]) +'_run_'+str(run_it+1)+name[name_type:], 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights('json_'+ name[:name_size+len('size_')]+ str(hidden_size[hidd_size]) +'_run_'+str(run_it+1)+name[name_type:] + '.h5')
		
		#defines a class to calculate the predicted MSE for each leg and separated for reachable and nonreachable points
		class mse_reach():
			def __init__(self):
				self.mse_reach_r1 = []	
				self.mse_nonreach_r1 = []
				self.mse_reach_r2 = []
				self.mse_nonreach_r2 = []
				self.mse_reach_r3 = []
				self.mse_nonreach_r3 = []
				self.mse_reach_r4 = []
				self.mse_nonreach_r4 = []
				self.mse_reach_r5 = []
				self.mse_nonreach_r5 = []
					
				reachable_receiver1 = np.squeeze(np.where(all_data_set_bin[:,6] == 1))
				nonreachable_receiver1 = np.squeeze(np.where(all_data_set_bin[:,6] == 0))
				reachable_receiver2 = np.squeeze(np.where(all_data_set_bin[:,10] == 1))
				nonreachable_receiver2 = np.squeeze(np.where(all_data_set_bin[:,10] == 0))
				reachable_receiver3 = np.squeeze(np.where(all_data_set_bin[:,14] == 1))
				nonreachable_receiver3 = np.squeeze(np.where(all_data_set_bin[:,14] == 0))
				reachable_receiver4 = np.squeeze(np.where(all_data_set_bin[:,18] == 1))
				nonreachable_receiver4 = np.squeeze(np.where(all_data_set_bin[:,18] == 0))
				reachable_receiver5 = np.squeeze(np.where(all_data_set_bin[:,22] == 1))
				nonreachable_receiver5 = np.squeeze(np.where(all_data_set_bin[:,22] == 0))

				#predict seperately reachable, nonreachable
				score_reach_r1 = loaded_model.predict(all_data_set[reachable_receiver1,0:3], verbose = 1)
				score_nonreach_r1 = loaded_model.predict(all_data_set[nonreachable_receiver1,0:3], verbose = 1)
				score_reach_r2 = loaded_model.predict(all_data_set[reachable_receiver2,0:3], verbose = 1)
				score_nonreach_r2 = loaded_model.predict(all_data_set[nonreachable_receiver2,0:3], verbose = 1)
				score_reach_r3 = loaded_model.predict(all_data_set[reachable_receiver3,0:3], verbose = 1)
				score_nonreach_r3 = loaded_model.predict(all_data_set[nonreachable_receiver3,0:3], verbose = 1)
				score_reach_r4 = loaded_model.predict(all_data_set[reachable_receiver4,0:3], verbose = 1)
				score_nonreach_r4 = loaded_model.predict(all_data_set[nonreachable_receiver4,0:3], verbose = 1)
				score_reach_r5 = loaded_model.predict(all_data_set[reachable_receiver5,0:3], verbose = 1)
				score_nonreach_r5 = loaded_model.predict(all_data_set[nonreachable_receiver5,0:3], verbose = 1)

				#calculate MSE of reachable and non reachable points
				reach_r1 = mean_squared_error(all_data_set[reachable_receiver1,3:6]*180, score_reach_r1[0][:,0:3]*180)
				nonreach_r1 = mean_squared_error(all_data_set[nonreachable_receiver1,3:6]*180, score_nonreach_r1[0][:,0:3]*180)
				reach_r2 = mean_squared_error(all_data_set[reachable_receiver2,7:10]*180, score_reach_r2[1][:,0:3]*180)
				nonreach_r2 = mean_squared_error(all_data_set[nonreachable_receiver2,7:10]*180, score_nonreach_r2[1][:,0:3]*180)
				reach_r3 = mean_squared_error(all_data_set[reachable_receiver3,11:14]*180, score_reach_r3[2][:,0:3]*180)
				nonreach_r3 = mean_squared_error(all_data_set[nonreachable_receiver3,11:14]*180, score_nonreach_r3[2][:,0:3]*180)
				reach_r4 = mean_squared_error(all_data_set[reachable_receiver4,15:18]*180, score_reach_r4[3][:,0:3]*180)
				nonreach_r4 = mean_squared_error(all_data_set[nonreachable_receiver4,15:18]*180, score_nonreach_r4[3][:,0:3]*180)
				reach_r5 = mean_squared_error(all_data_set[reachable_receiver5,19:22]*180, score_reach_r5[4][:,0:3]*180)
				nonreach_r5 = mean_squared_error(all_data_set[nonreachable_receiver5,19:22]*180, score_nonreach_r5[4][:,0:3]*180)
				
				self.mse_reach_r1.append(reach_r1)
				self.mse_nonreach_r1.append(nonreach_r1)
				self.mse_reach_r2.append(reach_r2)
				self.mse_nonreach_r2.append(nonreach_r2)
				self.mse_reach_r3.append(reach_r3)
				self.mse_nonreach_r3.append(nonreach_r3)
				self.mse_reach_r4.append(reach_r4)
				self.mse_nonreach_r4.append(nonreach_r4)
				self.mse_reach_r5.append(reach_r5)
				self.mse_nonreach_r5.append(nonreach_r5)
		
		mse_all = mse_reach()
		# Compress the losses into a dictionary
		pred = [mse_all.mse_reach_r1, mse_all.mse_nonreach_r1, mse_all.mse_reach_r2, mse_all.mse_nonreach_r2, mse_all.mse_reach_r3,
		mse_all.mse_nonreach_r3, mse_all.mse_reach_r4, mse_all.mse_nonreach_r4, mse_all.mse_reach_r5, mse_all.mse_nonreach_r5]
		dictionary = dict(zip(receivers, pred))
		mse_list[-1].append(dictionary)

#for plotting the exact same as before for plotting the MSE over the architecture
class plotting_vals():
	def __init__(self, loss):
		self.loss = loss
		# Construct target arrays for the 3D surface visualisation data
		# Single time series (up to 2000)
		vis_epochs = np.arange(0, len(mse_list[0][0][loss]))
		# Number of variations of architectures
		vis_hid_var = np.arange(0,len(mse_list))
		# Construct meshgrid of the two different arrays
		# = three 2 dimensional arrays, vis_X and vis_Y define regular locations in two dimensional space
		#   while vis_val_loss provides the specific values
		self.vis_X, self.vis_Y = np.meshgrid(vis_epochs, vis_hid_var)
		self.vis_val_loss = np.zeros(self.vis_X.shape)

		# Pushing the loaded data into this grid structure (vis_val_loss)
		# For this: the mean over the multiple runs for a single architecture is first calculated
		for arch_n in range(0,len(mse_list)):
			copied_val_loss = np.zeros((len(mse_list[arch_n]), len(mse_list[arch_n][0][loss]) ))
			for diff_runs in range(0, len(mse_list[arch_n])):
				copied_val_loss[diff_runs] = mse_list[arch_n][diff_runs][loss]
			self.vis_val_loss[arch_n] = np.log(np.mean(copied_val_loss, axis=0)) # np.log(np.array(hist_list[i]['val_loss']))
			#vis_val_loss[i][vis_val_loss[i]>20]= 20
			
		# Construct data at end of training
		copied_arch_val_loss = np.zeros((len(mse_list), len(mse_list[0])))
		for arch_n in range(0,len(mse_list)):
			for diff_runs in range(0, len(mse_list[arch_n])):
				# Getting the last loss - at end of training
				copied_arch_val_loss[arch_n][diff_runs] = mse_list[arch_n][diff_runs][loss][-1]
		self.mean_arch_val_loss = np.mean(copied_arch_val_loss, axis=1)
		#print(copied_arch_val_loss[2])
		self.std_arch_val_loss = np.std(copied_arch_val_loss, axis=1)
		self.arch_val_loss_lower_std = self.mean_arch_val_loss - self.std_arch_val_loss
		self.arch_val_loss_upper_std = self.mean_arch_val_loss + self.std_arch_val_loss 

receiver1_reach = plotting_vals('reach_r1')
receiver2_reach = plotting_vals('reach_r2')
receiver3_reach = plotting_vals('reach_r3')
receiver4_reach = plotting_vals('reach_r4')
receiver5_reach = plotting_vals('reach_r5') 
receiver1_nonreach = plotting_vals('nonreach_r1')
receiver2_nonreach = plotting_vals('nonreach_r2')
receiver3_nonreach = plotting_vals('nonreach_r3')
receiver4_nonreach = plotting_vals('nonreach_r4')
receiver5_nonreach = plotting_vals('nonreach_r5') 

#############################################################
#Different Architectures after Training for reachable points#
#############################################################
fig = plt.figure(figsize=(10, 8))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False)  
ax_arch.set_yscale('log')
ax_arch.set_xlim(-1, 9)  
ax_arch.set_xticks(np.arange(0,len(mse_list)))
ax_arch.set_xticklabels(['No hid.','1','2','4','8','16','32','64','128'])

# Use matplotlib's fill_between() call to create error bars.
plt.fill_between(range(0,len(receiver1_reach.mean_arch_val_loss)), receiver1_reach.arch_val_loss_lower_std,  
                 receiver1_reach.arch_val_loss_upper_std, color='g', alpha=0.5) 

plt.fill_between(range(0,len(receiver2_reach.mean_arch_val_loss)), receiver2_reach.arch_val_loss_lower_std,  
                 receiver2_reach.arch_val_loss_upper_std, color='r', alpha=0.5) 
                 
plt.fill_between(range(0,len(receiver3_reach.mean_arch_val_loss)), receiver3_reach.arch_val_loss_lower_std,  
                 receiver3_reach.arch_val_loss_upper_std, color='y', alpha=0.5) 
   
plt.fill_between(range(0,len(receiver4_reach.mean_arch_val_loss)), receiver4_reach.arch_val_loss_lower_std,  
                 receiver4_reach.arch_val_loss_upper_std, color=tableau20[1], alpha=0.5) 
                 
plt.fill_between(range(0,len(receiver5_reach.mean_arch_val_loss)), receiver5_reach.arch_val_loss_lower_std,  
                 receiver5_reach.arch_val_loss_upper_std, color=tableau20[3], alpha=0.5) 

plt.plot(range(0,len(receiver1_reach.mean_arch_val_loss)), receiver1_reach.mean_arch_val_loss, color='g', lw=2, label=legs[0]+' output')
plt.plot(range(0,len(receiver2_reach.mean_arch_val_loss)), receiver2_reach.mean_arch_val_loss, color='r', lw=2, label=legs[1]+' output')
plt.plot(range(0,len(receiver3_reach.mean_arch_val_loss)), receiver3_reach.mean_arch_val_loss, color='y', lw=2, label=legs[2]+' output')
plt.plot(range(0,len(receiver4_reach.mean_arch_val_loss)), receiver4_reach.mean_arch_val_loss, color=tableau20[0], lw=2, label=legs[3]+' output')
plt.plot(range(0,len(receiver5_reach.mean_arch_val_loss)), receiver5_reach.mean_arch_val_loss, color=tableau20[2], lw=2, label=legs[4]+' output')
plt.legend(frameon = False, fontsize=16)
plt.plot([-1,8], [10,10], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,8], [1,1], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,8], [0.1,0.1], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,8], [100,100], '-', color='gray', lw=1, alpha=0.25)
ax_arch.set_xlabel('# Hidden units', fontsize=20, labelpad=17)
ax_arch.set_ylabel('MSE', fontsize=20)
plt.tick_params(labelsize=19)  
plt.ylim(10**(0), 10**3)
ax_arch.set_title('MSE for reachable points \n'+mapp, fontsize=25) 
plt.tick_params(labelsize=19)
fig.savefig('Figures/'+str(name)+'_reach_MSEArch', dpi=300, bbox_inches='tight')

#################################################################
#Different Architectures after Training for non-reachable points#
#################################################################
fig = plt.figure(figsize=(10, 8))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False)  
ax_arch.set_yscale('log')
ax_arch.set_xlim(-1, 9)  
ax_arch.set_xticks(np.arange(0,len(mse_list)))
ax_arch.set_xticklabels(['No hid.','1','2','4','8','16','32','64','128'])

# Use matplotlib's fill_between() call to create error bars.
plt.fill_between(range(0,len(receiver1_nonreach.mean_arch_val_loss)), receiver1_nonreach.arch_val_loss_lower_std,  
                 receiver1_nonreach.arch_val_loss_upper_std, color='g', alpha=0.5) 

plt.fill_between(range(0,len(receiver2_nonreach.mean_arch_val_loss)), receiver2_nonreach.arch_val_loss_lower_std,  
                 receiver2_nonreach.arch_val_loss_upper_std, color='r', alpha=0.5) 
                 
plt.fill_between(range(0,len(receiver3_nonreach.mean_arch_val_loss)), receiver3_nonreach.arch_val_loss_lower_std,  
                 receiver3_nonreach.arch_val_loss_upper_std, color='y', alpha=0.5) 
   
plt.fill_between(range(0,len(receiver4_nonreach.mean_arch_val_loss)), receiver4_nonreach.arch_val_loss_lower_std,  
                 receiver4_nonreach.arch_val_loss_upper_std, color=tableau20[1], alpha=0.5) 
                 
plt.fill_between(range(0,len(receiver5_nonreach.mean_arch_val_loss)), receiver5_nonreach.arch_val_loss_lower_std,  
                 receiver5_nonreach.arch_val_loss_upper_std, color=tableau20[3], alpha=0.5) 

plt.plot(range(0,len(receiver1_nonreach.mean_arch_val_loss)), receiver1_nonreach.mean_arch_val_loss, color='g', lw=2, label=legs[0]+' output')
plt.plot(range(0,len(receiver2_nonreach.mean_arch_val_loss)), receiver2_nonreach.mean_arch_val_loss, color='r', lw=2, label=legs[1]+' output')
plt.plot(range(0,len(receiver3_nonreach.mean_arch_val_loss)), receiver3_nonreach.mean_arch_val_loss, color='y', lw=2, label=legs[2]+' output')
plt.plot(range(0,len(receiver4_nonreach.mean_arch_val_loss)), receiver4_nonreach.mean_arch_val_loss, color=tableau20[0], lw=2, label=legs[3]+' output')
plt.plot(range(0,len(receiver5_nonreach.mean_arch_val_loss)), receiver5_nonreach.mean_arch_val_loss, color=tableau20[2], lw=2, label=legs[4]+' output')
plt.legend(frameon = False, fontsize=16)
plt.plot([-1,8], [10,10], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,8], [1,1], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,8], [0.1,0.1], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,8], [100,100], '-', color='gray', lw=1, alpha=0.25)
ax_arch.set_xlabel('# Hidden units', fontsize=20, labelpad=17)
ax_arch.set_ylabel('MSE', fontsize=20)
plt.tick_params(labelsize=19)  
plt.ylim(10**(0), 10**3)
ax_arch.set_title('MSE for non-reachable points \n'+mapp, fontsize=25) 
plt.tick_params(labelsize=19)
fig.savefig('Figures/'+str(name)+'_nonreach_MSEArch', dpi=300, bbox_inches='tight')

########Print end vals###############
print(str(legs[0])+' end val loss nonreach: '+str(receiver1_nonreach.mean_arch_val_loss[-1]))
print(str(legs[1])+' end val loss nonreach: '+str(receiver2_nonreach.mean_arch_val_loss[-1]))
print(str(legs[2])+' end val loss nonreach: '+str(receiver3_nonreach.mean_arch_val_loss[-1]))
print(str(legs[3])+' end val loss nonreach: '+str(receiver4_nonreach.mean_arch_val_loss[-1]))
print(str(legs[4])+' end val loss nonreach: '+str(receiver5_nonreach.mean_arch_val_loss[-1]))

print(str(legs[0])+' end val loss reach: '+str(receiver1_reach.mean_arch_val_loss[-1]))
print(str(legs[1])+' end val loss reach: '+str(receiver2_reach.mean_arch_val_loss[-1]))
print(str(legs[2])+' end val loss reach: '+str(receiver3_reach.mean_arch_val_loss[-1]))
print(str(legs[3])+' end val loss reach: '+str(receiver4_reach.mean_arch_val_loss[-1]))
print(str(legs[4])+' end val loss reach: '+str(receiver5_reach.mean_arch_val_loss[-1]))

###############################################
# 2 A - Visualize generalization for large NN #
###############################################
# For the 128 hidden neurons:
# Show mean time series of val_loss and trainings loss
# = illustrates that there is no overfitting
class plotting_loss():
	def __init__(self, loss, val_loss):
		self.loss = loss
		self.val_loss = val_loss
		copied_val_loss = np.zeros((len(hist_list[-1]), len(hist_list[-1][0][val_loss]) ))

		for diff_runs in range(0, len(hist_list[-1])):
			copied_val_loss[diff_runs] = hist_list[-1][diff_runs][val_loss]
		self.mean_val_loss = np.mean(copied_val_loss, axis=0)
		self.std_val_loss = np.std(copied_val_loss, axis=0)
		self.val_loss_lower_std = self.mean_val_loss - self.std_val_loss
		self.val_loss_upper_std = self.mean_val_loss + self.std_val_loss

		copied_loss = np.zeros((len(hist_list[-1]), len(hist_list[-1][0][loss]) ))
		for diff_runs in range(0, len(hist_list[-1])):
			copied_loss[diff_runs] = hist_list[-1][diff_runs][loss]
		self.mean_loss = np.mean(copied_loss, axis=0)
		self.std_loss = np.std(copied_loss, axis=0)
		self.loss_lower_std = self.mean_loss - self.std_loss
		self.loss_upper_std = self.mean_loss + self.std_loss

l_receiver1 = plotting_loss(legs[0]+'_loss','val_'+legs[0]+'_loss')
l_receiver2 = plotting_loss(legs[1]+'_loss','val_'+legs[1]+'_loss')
l_receiver3 = plotting_loss(legs[2]+'_loss','val_'+legs[2]+'_loss')
l_receiver4 = plotting_loss(legs[3]+'_loss','val_'+legs[3]+'_loss')
l_receiver5 = plotting_loss(legs[4]+'_loss','val_'+legs[4]+'_loss')

#####################################################
# 2 B - Draw figure showing training and test error #
#####################################################
fig = plt.figure(figsize=(10,8))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_general = plt.subplot(111)  
ax_general.spines["top"].set_visible(False)  
ax_general.spines["right"].set_visible(False)  
ax_general.set_yscale('log')

# Use matplotlib's fill_between() call to create error bars.    
plt.fill_between(range(0,len(l_receiver1.mean_loss)), l_receiver1.loss_lower_std,  
                 l_receiver1.loss_upper_std, color='Crimson', alpha=0.5) 
plt.fill_between(range(0,len(l_receiver1.mean_val_loss)), l_receiver1.val_loss_lower_std,  
                 l_receiver1.val_loss_upper_std, color='OrangeRed', alpha=0.5) 

plt.fill_between(range(0,len(l_receiver2.mean_loss)), l_receiver2.loss_lower_std,  
                 l_receiver2.loss_upper_std, color='DarkCyan', alpha=0.5) 
plt.fill_between(range(0,len(l_receiver2.mean_val_loss)), l_receiver2.val_loss_lower_std,  
                 l_receiver2.val_loss_upper_std, color='DarkViolet', alpha=0.5)
                 
plt.fill_between(range(0,len(l_receiver3.mean_loss)), l_receiver3.loss_lower_std,  
                l_receiver3.loss_upper_std, color='SteelBlue', alpha=0.5) 
plt.fill_between(range(0,len(l_receiver3.mean_val_loss)), l_receiver3.val_loss_lower_std,  
                 l_receiver3.val_loss_upper_std, color='DarkKhaki', alpha=0.5)
                 
plt.fill_between(range(0,len(l_receiver4.mean_loss)), l_receiver4.loss_lower_std,  
                 l_receiver4.loss_upper_std, color='DarkGreen', alpha=0.5) 
plt.fill_between(range(0,len(l_receiver4.mean_val_loss)), l_receiver4.val_loss_lower_std,  
                 l_receiver4.val_loss_upper_std, color='DeepPink', alpha=0.5)
                 
plt.fill_between(range(0,len(l_receiver5.mean_loss)), l_receiver5.loss_lower_std,  
                 l_receiver5.loss_upper_std, color='SeaGreen', alpha=0.5) 
plt.fill_between(range(0,len(l_receiver5.mean_val_loss)), l_receiver5.val_loss_lower_std,  
                 l_receiver5.val_loss_upper_std, color='LimeGreen', alpha=0.5)

plt.plot(range(0,len(l_receiver1.mean_loss)), l_receiver1.mean_loss, color='Crimson', lw=1, label='Training '+ legs[0] + ' ' +str('{0:.3f}'.format(l_receiver1.mean_loss[-1])))
plt.plot(range(0,len(l_receiver1.mean_val_loss)), l_receiver1.mean_val_loss, color='OrangeRed', lw=1, label='Test '+ legs[0] + ' ' +str('{0:.3f}'.format(l_receiver1.mean_val_loss[-1])))

plt.plot(range(0,len(l_receiver2.mean_loss)), l_receiver2.mean_loss, color='DarkCyan', lw=1, label='Training '+ legs[1] + ' ' +str('{0:.3f}'.format(l_receiver2.mean_loss[-1])))
plt.plot(range(0,len(l_receiver2.mean_val_loss)), l_receiver2.mean_val_loss, color='DarkViolet', lw=1, label='Test '+ legs[1] + ' ' +str('{0:.3f}'.format(l_receiver2.mean_val_loss[-1])))

plt.plot(range(0,len(l_receiver3.mean_loss)), l_receiver3.mean_loss, color='SteelBlue', lw=1, label='Training '+ legs[2] + ' ' +str('{0:.3f}'.format(l_receiver3.mean_loss[-1])))
plt.plot(range(0,len(l_receiver3.mean_val_loss)), l_receiver3.mean_val_loss, color='DarkKhaki', lw=1, label='Test '+ legs[2] + ' ' +str('{0:.3f}'.format(l_receiver3.mean_val_loss[-1])))

plt.plot(range(0,len(l_receiver4.mean_loss)), l_receiver4.mean_loss, color='DarkGreen', lw=1, label='Training '+ legs[3] + ' ' +str('{0:.3f}'.format(l_receiver4.mean_loss[-1])))
plt.plot(range(0,len(l_receiver4.mean_val_loss)), l_receiver4.mean_val_loss, color='DeepPink', lw=1, label='Test '+ legs[3] + ' ' +str('{0:.3f}'.format(l_receiver4.mean_val_loss[-1])))

plt.plot(range(0,len(l_receiver5.mean_loss)), l_receiver5.mean_loss, color='SeaGreen', lw=1, label='Training '+ legs[4] + ' ' +str('{0:.3f}'.format(l_receiver5.mean_loss[-1])))
plt.plot(range(0,len(l_receiver5.mean_val_loss)), l_receiver5.mean_val_loss, color='LimeGreen', lw=1, label='Test '+ legs[4] + ' ' +str('{0:.3f}'.format(l_receiver5.mean_val_loss[-1])))

'''
plt.text(800, L2.mean_loss[-1]-2, 'L2 train', size=10)
plt.text(800, L2.mean_val_loss[-1]-1, 'L2 test', size=10)

plt.text(800, L3.mean_loss[-1]-1, 'L3 train', size=10)
plt.text(800, L3.mean_val_loss[-1]-0.5, 'L3 test', size=10)

plt.text(800, R1.mean_loss[-1]-2, 'R1 train', size=10)
plt.text(800, R1.mean_val_loss[-1]-1, 'R1 test', size=10)

plt.text(800, R2.mean_loss[-1]+5, 'R2 train', size=10)
plt.text(800, R2.mean_val_loss[-1]+8, 'R2 test', size=10)

plt.text(800, R3.mean_loss[-1]-1, 'R3 train', size=10)
plt.text(800, R3.mean_val_loss[-1]-0.5, 'R3 test', size=10)'''

plt.legend(frameon = False, fontsize=16)#plt.legend(loc='center', bbox_to_anchor=(0.5, -0.1), ncol=5) for the box to be below and outside of the plot
plt.title('Learning for 128 Hidden Neurons \n'+mapp, fontsize=25)
ax_arch.set_xlabel('Epoch', fontsize=20, labelpad=17)
ax_arch.set_ylabel('MSE', fontsize=20)
plt.tick_params(labelsize=19) 
plt.ylim(10**(0), 200)
fig.savefig('Figures/'+str(name)+'_reach_MSEEp128', dpi=300, bbox_inches='tight')

###############################################
###Evaluate binary info via confusion matrix###
###############################################
#reads and loads the model architecture/weights files
json_file = open('json_'+ name[:name_size+len('size_')]+ str(32)+'_run_'+str(5)+name[name_type:], 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('json_'+ name[:name_size+len('size_')]+ str(32)+'_run_'+str(5)+name[name_type:] + '.h5')

class confusion():
	def __init__(self, rec, column):
		self.rec = rec
		self.column = column
		y_pred = loaded_model.predict(all_data_set[:,0:3])
		y_pred_r = (y_pred[self.rec][:,3] > 0.5)
		self.cm = confusion_matrix(all_data_set_bin[:,self.column], y_pred_r)
		self.acc = (self.cm[0,0]+self.cm[1,1])/np.float(np.sum(self.cm))
		self.f1 = (2.*self.cm[0,0])/ ((2.*self.cm[0,0])+self.cm[0,1]+self.cm[1,0])
		self.mcc = matthews_corrcoef(all_data_set_bin[:,self.column], y_pred_r)
conf_r1 = confusion(0, 6)
conf_r2 = confusion(1, 10)
conf_r3 = confusion(2, 14)
conf_r4 = confusion(3, 18)
conf_r5 = confusion(4, 22)

print(legs[0]+' confusion matrix: ' + str(conf_r1.cm))
print(legs[0]+' accuracy: ' + str(conf_r1.acc))
print(legs[0]+' F1-score: ' + str(conf_r1.f1))
print(legs[0]+' MCC: ' + str(conf_r1.mcc))
print(legs[1]+' confusion matrix: ' + str(conf_r2.cm))
print(legs[1]+' accuracy: ' + str(conf_r2.acc))
print(legs[1]+' F1-score: ' + str(conf_r2.f1))
print(legs[1]+' MCC: ' + str(conf_r2.mcc))
print(legs[2]+' confusion matrix: ' + str(conf_r3.cm))
print(legs[2]+' accuracy: ' + str(conf_r3.acc))
print(legs[2]+' F1-score: ' + str(conf_r3.f1))
print(legs[2]+' MCC: ' + str(conf_r3.mcc))
print(legs[3]+' confusion matrix: ' + str(conf_r4.cm))
print(legs[3]+' accuracy: ' + str(conf_r4.acc))
print(legs[3]+' F1-score: ' + str(conf_r4.f1))
print(legs[3]+' MCC: ' + str(conf_r4.mcc))
print(legs[4]+' confusion matrix: ' + str(conf_r5.cm))
print(legs[4]+' accuracy: ' + str(conf_r5.acc))
print(legs[4]+' F1-score: ' + str(conf_r5.f1))
print(legs[4]+' MCC: ' + str(conf_r5.mcc))

#####################################################
##########Show all the plots generated###############
#####################################################
plt.show()
