##########################################################################################################
#####Script to visualize the loss of a given neural network with bidirectional leg pair mappings##########
##########################################################################################################
#Last update: 10.09.19
##########################################################################################################
#Part of the Master Thesis: Transformation of Spatial Contact Information among Limbs 
#by Nicole Walker, 2019
#This script visualizes the training history loss of a given network for either ipsilateral or
#contralateral mappings of leg pairs. It takes the forward and backward mapping or the left-to-right 
#and right-to-left mapping as input und plots them in comparison. It compares the loss of over different 
#architectures and over epochs for a single architecture.
########################################################################
#edit Nicole Walker
###########################################################################################################
###########################################################################################################
#Requirements: training histories of the neural networks (bidirectional) 
#					saved in the form: trainHistoryDict_5runs_2500ep_L1R1_
#			   Regression.py
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
from Regression import regr

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

#############################################
#####GUI to import training history file#####
#############################################
#Forward/Left-to-right mapping
root = Tk()
root.filename = tkFileDialog.askopenfilename()
print(root.filename)
with open(root.filename, 'rb') as file_pi:
    calc_hist_list = pickle.load(file_pi) 

#Backward/Right-to-left mapping
root2 = Tk()
root2.filename = tkFileDialog.askopenfilename()
print(root2.filename)
with open(root2.filename, 'rb') as file_pi:
    calc_back_hist_list = pickle.load(file_pi) 

#For automatizing the plot labeling
if 'L1R1' in root.filename:
	mapp = 'Left/Right Front Leg Mapping'
	legpair = 'L1R1'
	forw_label = 'left-to-right'
	back_label = 'right-to-left'
	dataset = 'Overlap_Targets_And_JointAngles_05_tar.mat'
elif 'L2R2' in root.filename:
	mapp = 'Left/Right Middle Leg Mapping'
	legpair = 'L2R2'
	forw_label = 'left-to-right'
	back_label = 'right-to-left'
	dataset = 'Overlap_Targets_And_JointAngles_05_tar.mat'
elif'L3R3' in root.filename:
	mapp = 'Left/Right Hind Leg Mapping'
	legpair = 'L3R3'
	forw_label = 'left-to-right'
	back_label = 'right-to-left'
	dataset = 'Overlap_Targets_And_JointAngles_05_tar.mat'
elif 'L1L2' in root.filename:
	mapp = 'Left Front-Middle Leg Mapping'
	legpair = 'L1L2'
	forw_label = 'front-to-back'
	back_label = 'back-to-front'
	dataset = 'Overlap_Targets_And_JointAngles_04_tar.mat'
elif 'L2L3' in root.filename:
	mapp = 'Left Middle-Hind Leg Mapping'
	legpair = 'L2L3'
	forw_label = 'front-to-back'
	back_label = 'back-to-front'
	dataset = 'Overlap_Targets_And_JointAngles_04_tar.mat'
elif 'R1R2' in root.filename:
	mapp = 'Right Front-Middle Leg Mapping'
	legpair = 'R1R2'
	forw_label = 'front-to-back'
	back_label = 'back-to-front'
	dataset = 'Overlap_Targets_And_JointAngles_04_tar.mat'
elif 'R2R3' in root.filename:
	mapp = 'Right Middle-Hind Leg Mapping'
	legpair = 'R2R3'
	forw_label = 'front-to-back'
	back_label = 'back-to-front'
	dataset = 'Overlap_Targets_And_JointAngles_04_tar.mat'

name_in = root.filename.index('Dict_')
name = root.filename[name_in+len('Dict_'):]
root.destroy()

name_in2 = root2.filename.index('Dict_')
name2 = root2.filename[name_in+len('Dict_'):]
root2.destroy()

##############################################
##########Define colors for plotting##########
##############################################
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
	def __init__(self, history):
		self.history = history
		# Construct target arrays for the 3D surface visualisation data
		# Single time series (up to 2000)
		vis_epochs = np.arange(0, len(history[0][0]['val_loss']))
		# Number of variations of architectures
		vis_hid_var = np.arange(0,len(history))
		# Construct meshgrid of the two different arrays
		# = three 2 dimensional arrays, vis_X and vis_Y define regular locations in two dimensional space
		#   while vis_val_loss provides the specific values
		self.vis_X, self.vis_Y = np.meshgrid(vis_epochs, vis_hid_var)
		self.vis_val_loss = np.zeros(self.vis_X.shape)

		# Pushing the loaded data into this grid structure (vis_val_loss)
		# For this: the mean over the multiple runs for a single architecture is first calculated
		for arch_n in range(0,len(history)):
			copied_val_loss = np.zeros((len(history[arch_n]), len(history[arch_n][0]['val_loss']) ))
			for diff_runs in range(0, len(history[arch_n])):
				copied_val_loss[diff_runs] = history[arch_n][diff_runs]['val_loss']
			self.vis_val_loss[arch_n] = np.log(np.mean(copied_val_loss, axis=0)) # np.log(np.array(hist_list[i]['val_loss']))
			#vis_val_loss[i][vis_val_loss[i]>20]= 20
			
		# Construct data at end of training
		copied_arch_val_loss = np.zeros((len(history), len(history[0])))
		for arch_n in range(0,len(history)):
			for diff_runs in range(0, len(history[arch_n])):
				# Getting the last loss - at end of training
				copied_arch_val_loss[arch_n][diff_runs] = history[arch_n][diff_runs]['val_loss'][-1]
		self.mean_arch_val_loss = np.mean(copied_arch_val_loss, axis=1)
		#print(copied_arch_val_loss[2])
		self.std_arch_val_loss = np.std(copied_arch_val_loss, axis=1)
		self.arch_val_loss_lower_std = self.mean_arch_val_loss - self.std_arch_val_loss
		self.arch_val_loss_upper_std = self.mean_arch_val_loss + self.std_arch_val_loss     

forw = plotting_vals(calc_hist_list)
backw = plotting_vals(calc_back_hist_list)	
'''
################################################
# 1 B - Draw 3D figure for the error over time #
################################################
fig = plt.figure(figsize=(10,8))
ax_3D = plt.subplot(111, projection='3d')    
ax_3D.set_yticklabels(['No Hidden','1','2','4','8','16','32','64','128'])
plt.ylim(0, 9)  

#surface plot (Combining C and D)  
surf = ax_3D.plot_surface(forw.vis_X, forw.vis_Y, forw.vis_val_loss, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.5, antialiased=False)
plt.gca().invert_yaxis()
fig.colorbar(surf, shrink=0.5, aspect=5)

ax_3D.set_xlabel('Epoch', fontsize=15, labelpad=18)
ax_3D.set_ylabel('# Hidden Units', fontsize=15, labelpad=18)
ax_3D.set_zlabel('log MSE', fontsize=15, labelpad=18)
plt.tick_params(labelsize=13)   
fig.savefig('Figures/'+str(name)+'_3D', dpi=300, bbox_inches='tight')
'''
###########################################################
# 1 C - Comparison different Architectures after Training #
###########################################################
fig = plt.figure(figsize=(10,8))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False)  
     
ax_arch.set_yscale('log')
ax_arch.set_xlim(-1, 9)  
ax_arch.set_xticks(np.arange(0,len(calc_hist_list)))
ax_arch.set_xticklabels(['No Hid.','1','2','4','8','16','32','64','128'])

# Use matplotlib's fill_between() call to create error bars.    
plt.fill_between(range(0,len(forw.mean_arch_val_loss)), forw.arch_val_loss_lower_std,  
                 forw.arch_val_loss_upper_std, color=tableau20[1], alpha=0.5) 
                 
plt.fill_between(range(0,len(backw.mean_arch_val_loss)), backw.arch_val_loss_lower_std,  
                 backw.arch_val_loss_upper_std, color=tableau20[3], alpha=0.5) 

#Regression lines
r = regr(legpair, 'forw', dataset)
r2 = regr(legpair, 'backw', dataset)

#Plotting
plt.plot(range(0,len(forw.mean_arch_val_loss)), forw.mean_arch_val_loss, color=tableau20[0], lw=2, label=forw_label)
plt.plot([-1,9], [r, r], '--', color='midnightblue', lw=2, label='Regression '+forw_label) #Squared error from Regression
plt.plot(range(0,len(backw.mean_arch_val_loss)), backw.mean_arch_val_loss, color=tableau20[2], lw=2, label=back_label)
plt.plot([-1,9], [r2, r2], '--', color=tableau20[6], lw=2, label='Regression '+back_label)
plt.plot([-1,9], [10,10], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,9], [1,1], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,9], [0.1,0.1], '-', color='gray', lw=1, alpha=0.25)
plt.plot([-1,9], [100,100], '-', color='gray', lw=1, alpha=0.25)
plt.legend(frameon=False, fontsize=16)
ax_arch.set_xlabel('# Hidden units', fontsize=20, labelpad=17)
ax_arch.set_ylabel('MSE', fontsize=20)
ax_arch.set_title(str(mapp), fontsize=25) 
plt.tick_params(labelsize=19)  
plt.ylim(10**(-2), 10**4)
#fig.savefig('Figures/'+str(name)+'_MSEArch', dpi=300, bbox_inches='tight')

###############################################
# 2 A - Visualize generalization for large NN #
###############################################
# For the 128 hidden neurons:
# Show mean time series of val_loss and trainings loss
# = illustrates that there is no overfitting

#Forward Testing loss
copied_val_loss = np.zeros((len(calc_hist_list[-1]), len(calc_hist_list[-1][0]['val_loss']) ))
for diff_runs in range(0, len(calc_hist_list[-1])):
    copied_val_loss[diff_runs] = calc_hist_list[-1][diff_runs]['val_loss']
mean_val_loss = np.mean(copied_val_loss, axis=0)
std_val_loss = np.std(copied_val_loss, axis=0)
val_loss_lower_std = mean_val_loss - std_val_loss
val_loss_upper_std = mean_val_loss + std_val_loss

#Forward Training loss
copied_loss = np.zeros((len(calc_hist_list[-1]), len(calc_hist_list[-1][0]['loss']) ))
for diff_runs in range(0, len(calc_hist_list[-1])):
    copied_loss[diff_runs] = calc_hist_list[-1][diff_runs]['loss']
mean_loss = np.mean(copied_loss, axis=0)
std_loss = np.std(copied_loss, axis=0)
loss_lower_std = mean_loss - std_loss
loss_upper_std = mean_loss + std_loss

#Backward Testing loss
copied_val_loss_back = np.zeros((len(calc_back_hist_list[-1]), len(calc_back_hist_list[-1][0]['val_loss']) ))
for diff_runs in range(0, len(calc_back_hist_list[-1])):
    copied_val_loss_back[diff_runs] = calc_back_hist_list[-1][diff_runs]['val_loss']
mean_val_loss_back = np.mean(copied_val_loss_back, axis=0)
std_val_loss_back = np.std(copied_val_loss_back, axis=0)
val_loss_lower_std_back = mean_val_loss_back - std_val_loss_back
val_loss_upper_std_back = mean_val_loss_back + std_val_loss_back

#Backward Training loss
copied_loss_back = np.zeros((len(calc_back_hist_list[-1]), len(calc_back_hist_list[-1][0]['loss']) ))
for diff_runs in range(0, len(calc_back_hist_list[-1])):
    copied_loss_back[diff_runs] = calc_back_hist_list[-1][diff_runs]['loss']
mean_loss_back = np.mean(copied_loss_back, axis=0)
std_loss_back = np.std(copied_loss_back, axis=0)
loss_lower_std_back = mean_loss_back - std_loss_back
loss_upper_std_back = mean_loss_back + std_loss_back

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
plt.fill_between(range(0,len(mean_loss)), loss_lower_std,  
                 loss_upper_std, color=tableau20[3], alpha=0.5) 
plt.fill_between(range(0,len(mean_val_loss)), val_loss_lower_std,  
                 val_loss_upper_std, color=tableau20[1], alpha=0.5) 

#Plotting Training and Test error
plt.plot(range(0,len(mean_loss)), mean_loss, color=tableau20[2], lw=1, label='Training')
plt.plot(range(0,len(mean_val_loss)), mean_val_loss, color=tableau20[0], lw=1, label='Test')
plt.legend(frameon=False, fontsize=16)

plt.title('Learning for 128 Hidden Neurons \n' + legpair, fontsize=20)
ax_general.set_xlabel('Epoch', fontsize=20, labelpad=17)
ax_general.set_ylabel('MSE (log)', fontsize=20) 
plt.tick_params(labelsize=19)
plt.ylim(10**(-2), 10**4)
#fig.savefig('Figures/'+str(name)+'_MSEEp128', dpi=300, bbox_inches='tight')

#Backward##################################################
fig = plt.figure(figsize=(10,8))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_general = plt.subplot(111)  
ax_general.spines["top"].set_visible(False)  
ax_general.spines["right"].set_visible(False)  
     
ax_general.set_yscale('log')

# Use matplotlib's fill_between() call to create error bars.    
plt.fill_between(range(0,len(mean_loss_back)), loss_lower_std_back,  
                 loss_upper_std_back, color=tableau20[3], alpha=0.5) 
plt.fill_between(range(0,len(mean_val_loss_back)), val_loss_lower_std_back,  
                 val_loss_upper_std_back, color=tableau20[1], alpha=0.5) 

#Plotting Training and Test error
plt.plot(range(0,len(mean_loss_back)), mean_loss_back, color=tableau20[2], lw=1, label='Training')
plt.plot(range(0,len(mean_val_loss_back)), mean_val_loss_back, color=tableau20[0], lw=1, label='Test')
plt.legend(frameon=False, fontsize=16)

plt.title('Learning for 128 Hidden Neurons \n' + legpair[2:4] + legpair[0:2], fontsize=20)
ax_general.set_xlabel('Epoch', fontsize=20, labelpad=17)
ax_general.set_ylabel('MSE (log)', fontsize=20) 
plt.tick_params(labelsize=19)
plt.ylim(10**(-2), 10**4)
#fig.savefig('Figures/'+str(name2)+'_MSEEp128', dpi=300, bbox_inches='tight')

##############################################################
#########Print end values of test and train###################
##############################################################
print('Forw mean loss: ' + str('{0:.3f}'.format(mean_loss[-1])))
print('Forw mean val loss: ' + str('{0:.3f}'.format(mean_val_loss[-1])))
print('Backw mean loss: ' + str('{0:.3f}'.format(mean_loss_back[-1])))
print('Backw mean val loss: ' + str('{0:.3f}'.format(mean_val_loss_back[-1])))

##############################################################
#############Show all plots generated########################
##############################################################
plt.show()
