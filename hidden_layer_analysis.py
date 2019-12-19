##########################################################################################################
####Script to analyze the learned representation of the network for seperate outputs of different legs####
##########################################################################################################
#Last update: 30.11.19
##########################################################################################################
#Part of the Master Thesis: Transformation of Spatial Contact Information among Limbs 
#by Nicole Walker, 2019
#This script is to analyze the hidden layer of a given neural network. It quantifies
#the directions is which the hidden nodes are sensitive via principal component analysis
#and maps the directional vectors to 2D mollweide projection. Vector statistics are applied.
#Eventually, the script calculates the angle between the mean output vector and the plane in which
#all PCs seem to lie.
####################################################
#edit Nicole Walker
###########################################################################################################
###########################################################################################################
#Requirements: used dataset with angles/targets mapping (e.g. Overlap_Targets_and_JointAngles_05_tar.mat)
#			   JSON files of the network model and weights
###########################################################################################################
###########################################################################################################

###########################################################
##########Importing packages and functions#################
###########################################################
import numpy as np
import keras
from keras import backend as bknd
from keras.models import Model
import json
from keras.models import model_from_json, load_model
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.pylab as py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.colors as allcolors
import pickle
import scipy.io
import scipy.linalg
from pylab import *
from theano	import function
from random import randrange
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Tkinter import Tk
import Tkinter, Tkconstants, tkFileDialog
from vector_iso import iso_stat
from error_map import error_map

########################################################
###########################load the file################
########################################################
#opens a GUI to select the file
root = Tk()
root.filename = tkFileDialog.askopenfilename()
print('Filename: ' + str(root.filename))
name_in = root.filename.index('json')
name = root.filename[name_in+len('json_'):]

#################################################
###########Parameters############################
#################################################
#The neuron of the network you want to look at
k = 5
#The leg you want the error map of
receiver = 'R1' 

#automatising the choice of senderleg data
if 'senderleg_L1' in root.filename:
	sender_leg = 0
	sender = 'L1'
elif 'senderleg_L2' in root.filename:
	sender_leg = 1
	sender = 'L2'
elif 'senderleg_L3' in root.filename:
	sender_leg = 2
	sender = 'L3'
elif 'senderleg_R1' in root.filename:
	sender_leg = 3
	sender = 'R1'
elif 'senderleg_R2' in root.filename:
	sender_leg = 4
	sender = 'R2'
elif 'senderleg_R3' in root.filename:
	sender_leg = 5
	sender = 'R3'
	
root.destroy()

#################################################
##########Data loading and preprocessing#########
#################################################
#Load the matlab files for the used data that is fed to the network

mat = scipy.io.loadmat('../Synthetic Data/synth_dataset_02.mat')

all_data_set = mat['ANGLES_map_yn']
all_target = mat['TARGET_map_clean']

all_target = all_target.squeeze()
all_target = all_target[sender_leg]
all_data_set = all_data_set.squeeze()
print(all_target.shape)
all_data_set = all_data_set[sender_leg]
only_angles = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21]
all_data_set[:,only_angles] /= 180


########################################################
########load the trained model and the weights##########
########################################################
#reads and loads the file
json_file = open(root.filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(root.filename + '.h5')

#get the activations of the hidden layer
get_activations = bknd.function([model.layers[0].input, bknd.learning_phase()], [model.layers[1].output])
weights = model.layers[1].get_weights()
activations = np.asarray(get_activations([all_data_set[:,0:3]]))
activations = np.squeeze(activations) 

#derivative of activations
deriv_act = activations[:,k]*(1-activations[:,k])

#weighs the input target points by the derivative
scaled = np.multiply(all_target[:,0:3], deriv_act.reshape((len(activations[:,k]),1)))

####Visualize the architecture of the model####
plot_model(model, to_file='./Figures/' + name +'_model_plot.png', show_shapes= True, show_layer_names=True)

#######################################################
#####Plotting of the tuning curves in 3D###############
#######################################################
#to facilitate plotting, the angles and target colums are separated in different variables
X, Y, Z = all_target[:,0], all_target[:,1], all_target[:,2]
L, M, N = all_data_set[:,0], all_data_set[:,1], all_data_set[:,2]

#defines the heatmap colors based on the activation
cmap = plt.get_cmap('viridis')
colmap = cm.ScalarMappable(cmap='viridis')
colmap.set_array(deriv_act)
norm=mpl.colors.Normalize(vmin=0, vmax=0.25)

colors =  deriv_act
#colors = colors[:,k]

#3D heatmap in carthesian space
fig = plt.figure(figsize=(13,11))   
ax_3D = plt.subplot(111, projection='3d')   

yg = ax_3D.scatter(X,Y,Z, cmap=cmap, c=colors, norm=norm,  marker='o')
cb = fig.colorbar(yg)
cb.ax.tick_params(labelsize=14)
plt.tick_params(labelsize=14)
ax_3D.view_init(azim=50+60, elev=20)
ax_3D.auto_scale_xyz([-30, 30], [-30,30], [-35,0])
ax_3D.set_xlabel('x-position of ' + sender + ' [mm]', fontsize=18, labelpad=20)
ax_3D.set_ylabel('y-position of ' + sender + ' [mm]', fontsize=18, labelpad=20)
ax_3D.set_zlabel('z-position of ' + sender + ' [mm]', fontsize=18, labelpad=20)
ax_3D.set_title('Derivative of Sigmoided Activation \n Neuron '+str(k), fontsize=22) 


#3D heatmap for angular space
fig = plt.figure(figsize=(10, 8))   
ax_3D = plt.subplot(111, projection='3d')   

yg = ax_3D.scatter(L,M,N, cmap=cmap, c=colors, norm=norm,  marker='o')
cb = fig.colorbar(yg)
plt.gca().invert_yaxis()
cb.ax.tick_params(labelsize=14)
plt.tick_params(labelsize=14)
ax_3D.set_xlabel('alpha of '+ sender + ' [$^\circ$]', fontsize=18, labelpad=20)
ax_3D.set_ylabel('beta of '+ sender+ ' [$^\circ$]', fontsize=18, labelpad=20)
ax_3D.set_zlabel('gamma of '+ sender+ ' [$^\circ$]', fontsize=18, labelpad=20)
ax_3D.set_title('Derivative of Sigmoided Activation \n Neuron '+str(k), fontsize=22) 

fig.savefig('PCA/Temp/' + name + '_carthesian_heatmap_activation', dpi=300, bbox_inches='tight')

############################################################
#############Principal Component Analysis###################
############################################################
#enter transform the scaled data
pca2 = PCA(n_components=3)
pca2.fit_transform(scaled)

###################################################
###############Plotting the PCA####################
###################################################
###Plot first and second PC###

figure = plt.figure(figsize=(8,8))
ax = plt.subplot(1,1,1)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('First/Second Component PCA')

yg = ax.scatter(princComp[:,0], princComp[:,1], cmap=cmap, c=colors, norm=norm,  marker='o')
cb = figure.colorbar(yg)
ax.grid()
figure.savefig('Figures/' + name + '_PCA_first_second_component', dpi=300, bbox_inches='tight')



print('Explained Variance Ratio: ' + str(pca2.explained_variance_ratio_))
print('Centered PCA Components: ' + str(pca2.components_))

###Mean of scaled targets for anchor point of the PCA vector###
mitt = np.mean(scaled, axis=0) 
print('Mean of scaled targets: ' + str(mitt))

###Centered Components Plot###
fig = plt.figure(figsize=(10, 8))   
ax2_3D = plt.subplot(111, projection='3d') 
ax2_3D.scatter(scaled[:,0], scaled[:,1], scaled[:,2], marker='o', alpha=0.02)
ax2_3D.quiver(mitt[0], mitt[1], mitt[2], pca2.components_[0][0],pca2.components_[0][1], pca2.components_[0][2], length=pca2.explained_variance_ratio_[0]*5, color='r', label='Comp 1: '+str(pca2.explained_variance_ratio_[0]))
ax2_3D.quiver(mitt[0], mitt[1], mitt[2], pca2.components_[1][0],pca2.components_[1][1], pca2.components_[1][2], length=pca2.explained_variance_ratio_[1]*5, color='b', label='Comp 2: '+str(pca2.explained_variance_ratio_[1]))
ax2_3D.quiver(mitt[0], mitt[1], mitt[2], pca2.components_[2][0],pca2.components_[2][1], pca2.components_[2][2], length=pca2.explained_variance_ratio_[2]*5, color='g', label='Comp 3: '+str(pca2.explained_variance_ratio_[2]))
ax2_3D.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3,fontsize=14)
plt.tick_params(labelsize=14)
cb.ax.tick_params(labelsize=14)
ax2_3D.set_xlabel('Weighted x-coordinates', fontsize=18, labelpad=20)
ax2_3D.set_ylabel('Weighted y-coordinates', fontsize=18, labelpad=20)
ax2_3D.set_zlabel('Weighted z-coordinates', fontsize=18, labelpad=20)
plt.title('Principal Component Analysis \n Neuron '+str(k), fontsize=22)
fig.savefig('PCA/Temp/' + name + '_PCA_centered', dpi=300, bbox_inches='tight')


ax2_3D.auto_scale_xyz([-7, 0], [-4,4], [-8,0]) #so we have all axes scaled equally
	
ax2_3D.view_init(azim=90, elev=0)
ax2_3D.set_xlabel('Weighted x-coordinates', fontsize=18, labelpad=20)
ax2_3D.set_ylabel('', fontsize=20, labelpad=17)
ax2_3D.set_zlabel('Weighted z-coordinates', fontsize=18, labelpad=20)
ax2_3D.get_legend().remove()
#plt.legend(fontsize=16, loc='center right')
fig.savefig('PCA/Temp/' + name + '_PCA_centered_xz', dpi=300, bbox_inches='tight')

ax2_3D.set_xlabel('', fontsize=20, labelpad=17)
ax2_3D.set_ylabel('Weighted y-coordinates', fontsize=18, labelpad=20)
ax2_3D.set_zlabel('Weighted z-coordinates',fontsize=18, labelpad=20)
ax2_3D.view_init(azim=0, elev=0)
fig.savefig('PCA/Temp/' + name + '_PCA_centered_yz', dpi=300, bbox_inches='tight')

ax2_3D.view_init(azim=0, elev=90)
ax2_3D.set_xlabel('Weighted x-coordinates', fontsize=18, labelpad=20)
ax2_3D.set_ylabel('Weighted y-coordinates', fontsize=18, labelpad=20)
ax2_3D.set_zlabel('', fontsize=20, labelpad=17)
#plt.legend(fontsize=16, loc='center left')
fig.savefig('PCA/Temp/' + name + '_PCA_centered_xy', dpi=300, bbox_inches='tight')




###3D heatmap with standardized and centered PCA vector in target carthesian space###
fig = plt.figure(figsize=(10, 8))   
ax_3D = plt.subplot(111, projection='3d')   

yg = ax_3D.scatter(X,Y,Z, cmap=cmap, c=colors, norm=norm,  marker='o', alpha=0.2)
ax_3D.quiver(0,0,0, pca.components_[0][0],pca.components_[0][1], pca.components_[0][2], length=5, color='r', label='z-transformed Comp 1: '+str(pca.explained_variance_ratio_[0]))
ax_3D.quiver(mitt[0], mitt[1], mitt[2], pca2.components_[0][0],pca2.components_[0][1], pca2.components_[0][2], length=5, color='b', label='not z-transformed Comp 1: '+str(pca2.explained_variance_ratio_[0]))
plt.legend(frameon=False, loc='center', bbox_to_anchor=(0.5, -0.05))
cb = fig.colorbar(yg)
plt.gca().invert_yaxis()

ax_3D.set_xlabel('x-position of ' + sender, fontsize=14)
ax_3D.set_ylabel('y-position of ' + sender, fontsize=14)
ax_3D.set_zlabel('z-position of ' + sender, fontsize=14)
ax_3D.set_title('Neuron '+str(k), fontsize=20) 
fig.savefig('Figures/' + name + '_PCA_vectors_Neuron' + str(k), dpi=300, bbox_inches='tight')


#########################################################
#######All neurons PCA vectors and Mollweide Plot########
#########################################################
#define two plots for all PCA vectors and resulting Mollweide
fig = plt.figure(figsize=(12, 10))   
ax2_3D = plt.subplot(111, projection='3d')

mode = 'cart'
#define the matrix for the angle between mean output vector and the repective plane
theta = np.zeros((5,6))
#matrix with all mean output vectors
mean_out = np.zeros((6,3))
mean_out[0,:] = np.array([-11.6008,0.5853, -20.2420])/scipy.linalg.norm(np.array([-11.6008,0.5853, -20.2420]))
mean_out[1,:] = np.array([-0.6057,2.7683, -20.2891])/scipy.linalg.norm(np.array([-0.6057,2.7683, -20.2891]))
mean_out[2,:] = np.array([9.3384,4.3838, -20.8747])/scipy.linalg.norm(np.array([9.3384,4.3838, -20.8747]))
mean_out[3,:] = np.array([-11.5743,-0.6158, -20.2406])/scipy.linalg.norm(np.array([-11.5743,-0.6158, -20.2406]))
mean_out[4,:] = np.array([-0.5784,-2.7908, -20.5304])/scipy.linalg.norm(np.array([-0.5784,-2.7908, -20.5304]))
mean_out[5,:] = np.array([9.3915,-4.4014, -20.9053])/scipy.linalg.norm(np.array([9.3915,-4.4014, -20.9053]))

boxplot_ml = np.zeros((5,6))
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'black']
legs_load = ['L1','L2','L3','R1','R2','R3']
for runs in range(1,6):
	for leg in range(0,6):
		directory = '../YesNo Network/'
		leg_name = 'json_5runs_1000ep_L1L2L3R1R2R3_senderleg_'+str(legs_load[leg])+'_size_128_run_'+str(runs)+'global_180tar_4outbin_0toEuc'
		json_file_PCA = open(directory+leg_name, 'r')
		loaded_model_json_PCA = json_file_PCA.read()
		json_file_PCA.close()
		model_PCA = model_from_json(loaded_model_json_PCA)
		get_activations_PCA = bknd.function([model_PCA.layers[0].input, bknd.learning_phase()], [model_PCA.layers[1].output])
		weights_PCA = model_PCA.layers[1].get_weights()
		model_PCA.load_weights(directory+leg_name + '.h5')
		all_data_set_PCA = mat['ANGLES_map_clean']
		all_target_PCA = mat['TARGET_map_clean']
		all_target_PCA = all_target_PCA.squeeze()
		all_target_PCA = all_target_PCA[leg]
		
		all_data_set_PCA = all_data_set_PCA.squeeze()
		all_data_set_PCA = all_data_set_PCA[leg]
		if runs == 1:
			all_data_set_PCA /= 180
			
		#modus
		if mode == 'cart':
			space = all_target_PCA[:,0:3]
		else:
			space = all_data_set_PCA[:,0:3]*180
		#get the sigmoided activations
		activations = np.asarray(get_activations_PCA([all_data_set_PCA[:,0:3]]))
		activations = np.squeeze(activations)

		#To store explained variance of all the neurons
		variance = np.zeros(activations.shape[1])
		x_components, y_components, z_components = np.zeros(activations.shape[1]), np.zeros(activations.shape[1]), np.zeros(activations.shape[1])
		all_scaled = []
		fig2 = plt.figure(figsize=(10, 10))  
		ax_moll = plt.subplot(111, projection = 'mollweide')
		if runs == 1:
			ax2_3D.plot([],[], color=colors[leg], label=legs_load[leg])
		########For each neuron extract the targets scaled by their sigmoided activation, the resulting PCA vectors and plot them#########
		for neuron in range(0,activations.shape[1]):
			#scaled = np.multiply(all_target[:,0:3],activations[:,neuron].reshape((len(activations[:,1]),1)))
			
			###this is if you use derivative scaled activation###
			deriv_act = activations[:,neuron]*(1-activations[:,neuron])
			scaled = np.multiply(space, deriv_act.reshape((len(activations[:,1]),1)))
			
			mitt = np.mean(scaled, axis=0)
			#print('mitt', mitt)
			pca = PCA(n_components=3)
			pca.fit_transform(scaled)
			#ax2_3D.quiver(mitt[0], mitt[1], mitt[2], pca.components_[0][0],pca.components_[0][1], pca.components_[0][2], length=pca.explained_variance_ratio_[0], color=colors[leg])#, alpha=0.1)
			#ax2_3D.scatter(mitt[0], mitt[1], mitt[2],color=colors[leg], alpha=0.2) 
			
			variance[neuron] = pca.explained_variance_ratio_[0]
			x_components[neuron] = pca.components_[0][0]
			y_components[neuron] = pca.components_[0][1]
			z_components[neuron] = pca.components_[0][2]
	
			all_scaled.append(scaled)
		
		####Mollweide Heatmap Plot####
		colmap.set_array(variance)
		#min is set to 0.33 because it's the least the first of the three components can have
		norm=mpl.colors.Normalize(vmin=0.33, vmax=1) 
		###calculate the angles from the PCA vectors####
		phi = np.arctan2(y_components, x_components) #azimuth
		psi = np.pi/2-np.arccos(z_components) #elevation
		phi_mean_out = np.arctan2(mean_out[leg,1], mean_out[leg,0])
		psi_mean_out = np.pi/2-np.arccos(mean_out[leg,2])
		#as PCA gives symmetry axis and not directions, it needs to be symmetrical
		anti_phi = np.zeros(len(phi))
		anti_phi = np.where(phi <= 0, phi+np.pi, phi-np.pi)
		anti_psi = -psi
		anti_phi_mean_out =np.where(phi_mean_out <= 0, phi_mean_out+np.pi, phi_mean_out-np.pi)
		anti_psi_mean_out = -psi_mean_out

		###Plot the end points of the PCA vectors on to a sphere to have a 2D representation###
		yg = ax_moll.scatter(phi, psi, cmap=cmap, c=variance, norm=norm,  marker='o')
		ax_moll.scatter(anti_phi, anti_psi, cmap=cmap, c=variance, norm=norm,  marker='o')
		ax_moll.scatter(phi_mean_out, psi_mean_out,  color='deeppink',  marker='o')
		ax_moll.scatter(anti_phi_mean_out, anti_psi_mean_out,  color='deeppink',  marker='o')
		cb = fig2.colorbar(yg)
		#Plotting setting for Mollweide Projection
		ax_moll.grid(True)
		plt.tick_params(labelsize=14)
		cb.ax.tick_params(labelsize=14)
		ax_moll.set_title('Mollweide Projection of Main Components \n (Cartesian) '+str(legs_load[leg]), fontsize=20)
		#fig2.savefig('PCA/Temp/' + leg_name + '_MollweideMeanOut_cart'+str(runs), dpi=300, bbox_inches='tight')
		plt.close()
		
		
		#######################################################################
		##################Test if the PCs are forming a plane##################
		#######################################################################
		#calculate the cross product beteen all PC vectors to see if they're aligned,
		#i.e. they're all normal vectors of the same plane
		norm_plane = np.zeros(((x_components.shape[0]*(x_components.shape[0]-1))/2, 3))
		count=0
		for i in range(0,x_components.shape[0]-1):
			for j in range(i+1, len(x_components)):
				cross_prod = np.cross(np.array([x_components[i], y_components[i], z_components[i]]),
				np.array([x_components[j], y_components[j], z_components[j]]))
				norm_plane[count,:] = cross_prod/scipy.linalg.norm(cross_prod)
				count = count +1
				
		mean_norm_plane = np.mean(norm_plane, axis=0)
		mean_norm_plane = mean_norm_plane/scipy.linalg.norm(np.copy(mean_norm_plane))
		
		
		#pick only one hemisphere by calculating the dot product and make them have the same sign
		for cross in range(0,norm_plane.shape[0]):
			if np.dot(norm_plane[cross,:],mean_norm_plane) < 0: #norm_plane[0,:]
				norm_plane[cross,:] = -np.copy(norm_plane[cross,:])

		#sum the normal vectors and divide them by the number of vectors.
		sum_norm = np.sum(norm_plane, axis = 0)
		len_sum_norm = scipy.linalg.norm(sum_norm)
		#mean the resultant length of the vector
		mean_len = len_sum_norm/norm_plane.shape[0]
		boxplot_ml[runs-1, leg] = mean_len
		print(str(legs_load[leg])+' Mean length: ' + str(mean_len))
		
		#calculate the angle between the plane and the mean ouput vector
		cos_theta = np.abs(np.dot(mean_norm_plane,mean_out[leg,:]))
		print(np.arccos(cos_theta))
		theta[runs-1, leg] = 90-np.degrees(np.arccos(cos_theta))
print('Theta angle between plane and mean output vector: ',theta)
print('Mean theta over 5 runs per leg: ',np.mean(theta, axis=0))

#Plotting mean output vectors
ax2_3D.quiver(0, 0, 0, -11.6008,0.5853, -20.2420, length=0.2, color='deeppink', label='mean output')
ax2_3D.quiver(0, 0, 0, -0.6057,2.7683, -20.2891, length=0.2, color='deeppink')
ax2_3D.quiver(0, 0, 0, 9.3384,4.3838, -20.8747, length=0.2, color='deeppink')
ax2_3D.quiver(0, 0, 0, -11.5743,-0.6158, -20.2406, length=0.2, color='deeppink')
ax2_3D.quiver(0, 0, 0, -0.5784,-2.7908, -20.5304, length=0.2, color='deeppink')
ax2_3D.quiver(0, 0, 0, 9.3915,-4.4014, -20.9053, length=0.2, color='deeppink')


#Axis label etc. divided whether Cartesian or angle space are plotted
if mode == 'cart':
	ax2_3D.set_xlabel('Weighted x-coordinates', fontsize=20, labelpad=17)
	ax2_3D.set_ylabel('Weighted y-coordinates', fontsize=20, labelpad=17)
	ax2_3D.set_zlabel('Weighted z-coordinates', fontsize=20, labelpad=17)
	ax2_3D.auto_scale_xyz([-3, 3], [-2.5,2.5], [-5,0]) #so we have all axes scaled equally
	ax2_3D.set_xticks(np.arange(-3,4, 1))
	ax2_3D.set_yticks(np.arange(-2.5,3.5, 1))
	plt.tick_params(labelsize=14)
	plt.legend(fontsize=16, loc='center left')

	ax2_3D.set_title('Main Component Directions (Cartesian)', fontsize=22)
	fig.savefig('PCA/Temp/' + leg_name + '_MainCompBase_all_cart', dpi=300, bbox_inches='tight')
	
	ax2_3D.view_init(azim=90, elev=0)
	ax2_3D.set_xlabel('Weighted x-coordinates', fontsize=20, labelpad=20)
	ax2_3D.set_ylabel('', fontsize=20, labelpad=17)
	ax2_3D.set_zlabel('Weighted z-coordinates', fontsize=20, labelpad=20)
	ax2_3D.get_legend().remove()
	ax2_3D.set_title('')
	#plt.legend(fontsize=16, loc='center right')
	fig.savefig('PCA/Temp/' + leg_name + '_MainCompBase_all_cart_xz', dpi=300, bbox_inches='tight')

	ax2_3D.set_xlabel('', fontsize=20, labelpad=17)
	ax2_3D.set_ylabel('Weighted y-coordinates', fontsize=20, labelpad=20)
	ax2_3D.set_zlabel('Weighted z-coordinates', fontsize=20, labelpad=20)
	ax2_3D.view_init(azim=0, elev=0)
	fig.savefig('PCA/Temp/' + leg_name + '_MainCompBase_all_cart_yz', dpi=300, bbox_inches='tight')

	ax2_3D.view_init(azim=0, elev=90)
	ax2_3D.set_xlabel('Weighted x-coordinates', fontsize=20, labelpad=20)
	ax2_3D.set_ylabel('Weighted y-coordinates', fontsize=20, labelpad=20)
	ax2_3D.set_zlabel('', fontsize=20, labelpad=17)
	#plt.legend(fontsize=16, loc='center left')
	fig.savefig('PCA/Temp/' + leg_name + '_MainCompBase_all_cart_xy', dpi=300, bbox_inches='tight')
	
else:
	#Plotting setting for PCA Vectors
	ax2_3D.set_xlabel('Weighted alpha angle', fontsize=20, labelpad=17)
	ax2_3D.set_ylabel('Weighted beta angle', fontsize=20, labelpad=17)
	ax2_3D.set_zlabel('Weighted gamma angle', fontsize=20, labelpad=17)
	ax2_3D.auto_scale_xyz([-9,9], [-4,2], [0,20])
	plt.tick_params(labelsize=14)
	plt.legend(fontsize=16, loc='center left')
	
	ax2_3D.set_title('Main Component Directions (Angles)', fontsize=22)
	fig.savefig('PCA//Temp/' + leg_name + '_MainCompDirDeriv_all_ang', dpi=300, bbox_inches='tight')
	
	ax2_3D.view_init(azim=90, elev=0)
	ax2_3D.set_xlabel('Weighted alpha angle', fontsize=20, labelpad=20)
	ax2_3D.set_ylabel('', fontsize=20, labelpad=17)
	ax2_3D.set_zlabel('Weighted gamma angle', fontsize=20, labelpad=20)
	#ax2_3D.get_legend().remove()
	ax2_3D.set_title('')
	fig.savefig('PCA//Temp/' + leg_name + '_MainCompDirDeriv_all_ang_ag', dpi=300, bbox_inches='tight')

	ax2_3D.set_xlabel('', fontsize=20, labelpad=17)
	ax2_3D.set_ylabel('Weighted beta angle', fontsize=20, labelpad=20)
	ax2_3D.set_zlabel('Weighted gamma angle', fontsize=20, labelpad=20)
	ax2_3D.view_init(azim=0, elev=0)
	fig.savefig('PCA//Temp/' + leg_name + '_MainCompDirDeriv_all_ang_bg', dpi=300, bbox_inches='tight')

	ax2_3D.view_init(azim=0, elev=90)
	ax2_3D.set_xlabel('Weighted alpha angle', fontsize=20, labelpad=20)
	ax2_3D.set_ylabel('Weighted beta angle', fontsize=20, labelpad=20)
	ax2_3D.set_zlabel('', fontsize=20, labelpad=17)
	#plt.legend(fontsize=16, loc='center left')
	fig.savefig('PCA/Temp/' + leg_name + '_MainCompDirDeriv_all_ang_ab', dpi=300, bbox_inches='tight')


#######################################################################
##################Test if the PCs are forming a plane##################
#######################################################################
#calculate the cross product beteen all PC vectors to see if they're aligned,
#i.e. they're all normal vectors of the same plane

ens=1000
vecs = np.random.normal(size=(ens, x_components.shape[0]*(x_components.shape[0]-1)/2,3))
norm_rand = np.zeros((ens, (x_components.shape[0]*(x_components.shape[0]-1))/2, 3))
sum_rand = np.zeros((ens,3))
len_sum_rand = np.zeros(ens)
mean_rand = np.zeros(ens)
count=np.zeros(ens, dtype=int)
for n in range(0,ens):
	for k in range(0,len(vecs[n,:,0])):
		vecs[n,k,:] = np.copy(vecs[n,k,:])/float(scipy.linalg.norm(vecs[n,k,:]))

	for i in range(0,x_components.shape[0]-1):
		for j in range(i+1, len(x_components)):
			cross_rand = np.cross(np.array([vecs[n,i,0], vecs[n,i,1], vecs[n,i,2]]),
			np.array([vecs[n,j,0], vecs[n,j,1], vecs[n,j,2]]))
			norm_rand[n,count[n],:] = cross_rand/scipy.linalg.norm(cross_rand)
			count[n] = count[n] +1

	#pick only one hemisphere by change sign according to the result of the dot product
		for cross in range(0,len(norm_rand[n,:,0])):
			if norm_rand[n,cross,2] < 0:
				norm_rand[n,cross,:] = -np.copy(norm_rand[n,cross,:])
	#sum the normal vectors and divide them by the number of vectors.
	sum_rand[n,:] = np.sum(norm_rand[n,:,:], axis=0)
	len_sum_rand[n] = scipy.linalg.norm(sum_rand[n,:])
	#mean the resultant length of the vector
	mean_rand[n] = len_sum_rand[n]/len(norm_rand[n,:,0])
	print(n)
#expectation value and sigmas
exp = np.mean(mean_rand)
mean_rand = np.sort(mean_rand)
sigma = np.zeros((3,2))
sigma[0,0] = exp - mean_rand[int(0.16*ens)-1]
sigma[1,0] = exp - mean_rand[int(0.02*ens)-1]
sigma[2,0] = exp - mean_rand[int(0.001*ens)-1]
sigma[0,1] = -exp + mean_rand[-int(0.16*ens)]
sigma[1,1] = -exp + mean_rand[-int(0.02*ens)]
sigma[2,1] = -exp + mean_rand[-int(0.001*ens)]
print('Expectation value and sigmas: ',exp, sigma)


#calculate scalar product betwen the normal vectors (but only first with the others, otherwise for 128 it would be 10^8 dot products)
dot_norm_plane = np.zeros((((x_components.shape[0]*(x_components.shape[0]-1))/2)-1))
for k in range(1, norm_plane.shape[0]):
	dot_norm_plane[k-1] = np.abs(np.dot(norm_plane[0,:], norm_plane[k,:]))

#mean them to see if they're close to 1
mean_dot_norm_plane = np.mean(dot_norm_plane)
print('Mean of dot product of PC plane norm vectors: ' + str(mean_dot_norm_plane))

#use the mean of the vectors as norm vector to calculate the plane
mean_norm_plane = np.mean(norm_plane, axis=0)
mean_norm_plane = mean_norm_plane/scipy.linalg.norm(mean_norm_plane)

#######################################################################
###################Boxplot of plane statistics#########################
#######################################################################
#Cartesian###############################################################
fig_boxcart = plt.figure(figsize=(10, 10))  
ax_boxcart = plt.subplot(111)
ax_boxcart.boxplot(boxplot_ml)
ax_boxcart.set_xticklabels(['L1','L2','L3','R1','R2','R3'])
ax_boxcart.set_title('Plane statistics for 128 nodes (Cartesian)', fontsize=20)
ax_boxcart.hlines(0.5042154664407631 ,0,7)
ax_boxcart.fill_between(range(0,8), 0.5042154664407631,0.5042154664407631+0.02493122, color='g', alpha=0.6) 
ax_boxcart.fill_between(range(0,8), 0.5042154664407631,0.5042154664407631+0.05151209, color='g', alpha=0.4) 
ax_boxcart.fill_between(range(0,8), 0.5042154664407631,0.5042154664407631+0.073613615, color='g', alpha=0.2) 
ax_boxcart.fill_between(range(0,8), 0.5042154664407631,0.5042154664407631-0.02493122, color='g', alpha=0.6) 
ax_boxcart.fill_between(range(0,8), 0.5042154664407631,0.5042154664407631-0.05151209, color='g', alpha=0.4) 
ax_boxcart.fill_between(range(0,8), 0.5042154664407631,0.5042154664407631-0.073613615, color='g', alpha=0.2) 
ax_boxcart.text(3.5, 0.515,r'$\sigma$', fontsize=13)       
ax_boxcart.text(3.5, 0.485,r'$\sigma$', fontsize=13)  
ax_boxcart.text(3.5, 0.535,r'$2\sigma$', fontsize=13)  
ax_boxcart.text(3.5, 0.46,r'$2\sigma$', fontsize=13)  
ax_boxcart.text(3.5, 0.56,r'$3\sigma$', fontsize=13)  
ax_boxcart.text(3.5, 0.435,r'$3\sigma$', fontsize=13) 
plt.tick_params(labelsize=15)
ax_boxcart.set_ylim(0.4,1)
ax_boxcart.set_xlabel('Leg', fontsize=15, labelpad=17)
ax_boxcart.set_ylabel('Test statistic', fontsize=16, labelpad=17)
fig_boxcart.savefig('PCA/planestats_128_cart_04', dpi=300, bbox_inches='tight') 

