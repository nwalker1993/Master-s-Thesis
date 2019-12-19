##########################################################################################################
########################Script plot leg postures from a single sample mapping#############################
##########################################################################################################
#Last update: 25.11.19
##########################################################################################################
#Part of the Master Thesis: Transformation of Spatial Contact Information among Limbs 
#by Nicole Walker, 2019
#This function takes at least a sample point, one sender and one receiver leg as input but any number of 
#receiver legs can be provided. It plots the forward kinemtatics of a given posture and the respective 
#targeting postures of the receivers provided. Additionally, the predictions of a trained network can be
#fed to the function such that it plots the predicted posture for comparison as well. 
########################################################################
#edit Nicole Walker
###########################################################################################################
###########################################################################################################
#Requirements: used dataset with angles/targets mapping (e.g. synth_dataset_03.mat)
#			  optional error_map.py
###########################################################################################################
###########################################################################################################

###############################################################
############Importing packages and functions###################
###############################################################

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.io
import pickle
from error_map import error_map

##############################################################
####################Define the function#######################
##############################################################
def plot_forwkin(sender_leg, rec_leg, **keyword_parameters):
	
	#sample keyword
	if 'N' in keyword_parameters:
		N = keyword_parameters['N']
	print(keyword_parameters)
		
	#load needed data	
	mat = scipy.io.loadmat('../Synthetic Data/synth_dataset_03.mat')
	all_data_set = mat['ANGLES_map_clean']
	all_target = mat['TARGET_map_clean']
	targets = all_target.squeeze()
	angle = all_data_set.squeeze()
	
	#definition of variables depending on sender and eceiver leg
	if sender_leg == 'L1':
		sender = 0
		col_send = range(0,3)
	elif sender_leg == 'L2':
		sender = 1
		col_send = range(3,6)
	elif sender_leg == 'L3':
		sender = 2
		col_send = range(6,9)
	elif sender_leg == 'R1':
		sender = 3
		col_send = range(9,12)
	elif sender_leg == 'R2':
		sender = 4
		col_send = range(12,15)
	elif sender_leg == 'R3':
		sender = 5
		col_send = range(15,18)

	if rec_leg == 'L1':
		rec = 0
		col_rec = range(0,3)
	elif rec_leg == 'L2':
		rec = 1
		col_rec = range(3,6)
	elif rec_leg == 'L3':
		rec = 2
		col_rec = range(6,9)
	elif rec_leg == 'R1':
		rec = 3
		col_rec = range(9,12)
	elif rec_leg == 'R2':
		rec = 4
		col_rec = range(12,15)
	elif rec_leg == 'R3':
		rec = 5
		col_rec = range(15,18)
		
	# plot the points depending on sender and receiver
	fig = plt.figure(figsize=(12,10))
	ax = fig.add_subplot(111, projection='3d')	
	class forwkin():
		def __init__(self, leg, angle, targets, color):
			self.leg = leg
			self.angle = angle
			self.targets = targets
			if self.leg == 'L1':
				length = [1.3, 16.6, 20.7]
				point = [0,0,0]
				shift = 0
			elif self.leg == 'L2':
				length = [1.4, 12.1, 16.1]
				point = [-18,1,0]
				shift = 0
			elif self.leg == 'L3':
				length = [1.5, 14.6, 19.4]
				point = [-30,1,0]
				shift = 0
			elif self.leg == 'R1':
				length  = [-1.3, -16.5, -20.8]
				point = [0,-2,0]
				shift = 180
			elif self.leg == 'R2':
				length = [-1.4, -12.2, -16.1]
				point = [-18,-3,0]
				shift = 180
			elif self.leg == 'R3':
				length = [-1.5, -14.6, -19.5]
				point = [-30,-3,0]
				shift = 180

			# unpack the first point
			self.x, self.y, self.z = point

			# find the end point for coxa
			self.cox_endy = self.y + length[0] * math.cos(math.radians(self.angle[0]))
			self.cox_endx = self.x - length[0] * math.sin(math.radians(self.angle[0]))
			self.cox_endz = self.z 

			#use end of coxa as start for fem
			self.fem_endy = np.copy(self.cox_endy) + length[1] * math.cos(math.radians(self.angle[0])) * math.cos(math.radians(self.angle[1]))
			self.fem_endx = np.copy(self.cox_endx) - length[1] * math.sin(math.radians(self.angle[0])) * math.cos(math.radians(self.angle[1]))
			self.fem_endz = np.copy(self.cox_endz) - length[1] * math.sin(math.radians(self.angle[1]+shift))

			#use fem as start for tib
			self.tib_endy = np.copy(self.targets[1]) + self.y
			self.tib_endx = np.copy(self.targets[0]) + self.x
			self.tib_endz = np.copy(self.targets[2]) + self.z
			
			#plot
			ax.plot([self.x, self.cox_endx], [self.y, self.cox_endy], [self.z, self.cox_endz], color=color)
			ax.plot([self.cox_endx, self.fem_endx], [self.cox_endy, self.fem_endy], [self.cox_endz, self.fem_endz], color=color)
			ax.plot([self.fem_endx, self.tib_endx], [self.fem_endy, self.tib_endy], [self.fem_endz, self.tib_endz], color=color)
			
			'''
			#print the end points of the limb parts
			print('Plotted angles',self.leg, angle)
			print('Leg', self.leg)
			print('origin', self.x, self.y, self.z)
			print('cox', self.cox_endx, self.cox_endy, self.cox_endz)
			print('fem', self.fem_endx, self.fem_endy, self.fem_endz)
			print('tib', self.tib_endx, self.tib_endy, self.tib_endz)
			'''
			
	#uses the plotting function per provided legs
	leg_1 = forwkin(sender_leg, angle[sender][N,col_send], targets[sender][N, col_send], color='orange')
	leg_2 = forwkin(rec_leg, angle[sender][N,col_rec], targets[sender][N, col_rec], color='yellow')
	#if the error_map.py option is wanted, it plots the predicted posture compared to the true posture
	if 'alt' in keyword_parameters:
		alt = keyword_parameters['alt']
		leg_alt = forwkin(rec_leg, alt.angles, targets[sender][N, col_rec], color='deeppink')
	#if more than one receiver is provided, it checks the keywords given to the function and plots the legs accordingly
	if 'rec2_leg' in keyword_parameters:
		rec2_leg = keyword_parameters['rec2_leg']
		if rec2_leg == 'L1':
			col_rec = range(0,3)
		elif rec2_leg == 'L2':
			col_rec = range(3,6)
		elif rec2_leg == 'L3':
			col_rec = range(6,9)
		elif rec2_leg == 'R1':
			col_rec = range(9,12)
		elif rec2_leg == 'R2':
			col_rec = range(12,15)
		elif rec2_leg == 'R3':
			col_rec = range(15,18)
		leg_3 = forwkin(rec2_leg, angle[sender][N,col_rec], targets[sender][N, col_rec], color='red')
	if 'rec3_leg' in keyword_parameters:
		rec3_leg = keyword_parameters['rec3_leg']
		if rec3_leg == 'L1':
			col_rec = range(0,3)
		elif rec3_leg == 'L2':
			col_rec = range(3,6)
		elif rec3_leg == 'L3':
			col_rec = range(6,9)
		elif rec3_leg == 'R1':
			col_rec = range(9,12)
		elif rec3_leg == 'R2':
			col_rec = range(12,15)
		elif rec3_leg == 'R3':
			col_rec = range(15,18)
		leg_4 = forwkin(rec3_leg, angle[sender][N,col_rec], targets[sender][N, col_rec], color='green')
	if 'rec4_leg' in keyword_parameters:
		rec4_leg = keyword_parameters['rec4_leg']
		if rec4_leg == 'L1':
			col_rec = range(0,3)
		elif rec4_leg == 'L2':
			col_rec = range(3,6)
		elif rec4_leg == 'L3':
			col_rec = range(6,9)
		elif rec4_leg == 'R1':
			col_rec = range(9,12)
		elif rec4_leg == 'R2':
			col_rec = range(12,15)
		elif rec4_leg == 'R3':
			col_rec = range(15,18)
		leg_5 = forwkin(rec4_leg, angle[sender][N,col_rec], targets[sender][N, col_rec], color='blue')
	if 'rec5_leg' in keyword_parameters:
		rec5_leg = keyword_parameters['rec5_leg']
		if rec5_leg == 'L1':
			col_rec = range(0,3)
		elif rec5_leg == 'L2':
			col_rec = range(3,6)
		elif rec5_leg == 'L3':
			col_rec = range(6,9)
		elif rec5_leg == 'R1':
			col_rec = range(9,12)
		elif rec5_leg == 'R2':
			col_rec = range(12,15)
		elif rec5_leg == 'R3':
			col_rec = range(15,18)
		leg_6 = forwkin(rec5_leg, angle[sender][N,col_rec], targets[sender][N, col_rec], color='black')


	#the scatter points give the head and thoracic-coxal joints
	ax.scatter(leg_1.targets[0], leg_1.targets[1], leg_1.targets[2])
	ax.scatter(-18,1,0) 
	ax.scatter(-30, 1, 0) 
	ax.scatter(0, 0, 0 )
	ax.scatter(-30, -3, 0) 
	ax.scatter(-18, -3, 0 )
	ax.scatter(0, -2, 0 )
	ax.scatter(7, -1, 0, s=150) #head
	ax.auto_scale_xyz([-50,10], [-20,20], [-40,5])
	#ax.scatter(scat[:,0]+18, scat[:,1], scat[:,2], alpha=0.04, c='b')#scat[:,0]+18 for first to second leg
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	'''
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	ax2.plot([leg_1.x, leg_1.cox_endx], [leg_1.y, leg_1.cox_endy])
	ax2.plot([leg_1.cox_endx, leg_1.fem_endx], [leg_1.cox_endy, leg_1.fem_endy])
	ax2.plot([leg_1.fem_endx, leg_1.tib_endx], [leg_1.fem_endy, leg_1.tib_endy])
	

	fig3 = plt.figure()
	ax3 = fig3.add_subplot(111)
	ax3.plot([leg_2.x, leg_2.cox_endx], [leg_2.y, leg_2.cox_endy])
	ax3.plot([leg_2.cox_endx, leg_2.fem_endx], [leg_2.cox_endy, leg_2.fem_endy])
	ax3.plot([leg_2.fem_endx, leg_2.tib_endx], [leg_2.fem_endy, leg_2.tib_endy])
	'''
	plt.show()

	

#network weights and architecture used for error_map.py to predict the postures by the network
model_file = '../YesNo Network/json_5runs_1000ep_L1L2L3R1R2R3_senderleg_L1_size_32_run_1global_180tar_4outbin_0toEuc'
N = 28638 #28638 #20000 #29998 #sample point number
sender_leg = 'L1'
rec_leg = 'R1'

#error_map.py
er_rec1, er_rec2, er_rec3, er_rec4, er_rec5 = error_map(sender_leg, model_file, N)
#plot_forwkin(sender_leg, rec_leg, N=N, alt=er_rec5)

#call plotleg.py function with prediction from error_map.py
plot_forwkin(sender_leg, rec_leg, rec2_leg='R2', rec3_leg='R1', rec4_leg='L2', rec5_leg='L3', N=N, alt=er_rec3)


