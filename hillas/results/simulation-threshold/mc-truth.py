# Create a master spreadsheet with each event from the simulation
# Contains information about each event's particle parameters and morphology/classification
# i.e. Monte-Carlo truth

import pandas as pd
import numpy as np
import os

data = pd.DataFrame()

source_dir = './sample_stats'

for particle_dir in os.listdir(source_dir):
	for file in os.listdir(source_dir + "/" + particle_dir):
	
		path = source_dir + "/" + particle_dir + "/" + file
		particle_type = particle_dir
		energy = file.split("_")[0]
		theta = file.split("_")[2]
		phi = file.split("_")[4]
		
		try:
			event_id_head = energy[:-3].split('.')[0] + '0' + energy[:-3].split('.')[1] + theta.split('.')[0] + theta.split('.')[1] + phi.split('.')[0] + phi.split('.')[1] 
		except IndexError:
			event_id_head = energy[:-3] + theta.split('.')[0] + theta.split('.')[1] + phi.split('.')[0] + phi.split('.')[1] 

		df = pd.read_csv(path)
		
		try:
			df.drop('x_coord')
			df.drop('y_coord')
		except KeyError:
			pass
			
		try:
			df.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
			df.drop(["a"], axis=1, inplace=True)
		except KeyError:
			pass
			
		try:
			df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
		except KeyError:
			pass
			
		try:
			df.drop(df.columns[df.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)
		except KeyError:
			pass
		
		df['Particle'] = particle_type
		df['Energy'] = energy
		df['theta'] = theta
		df['phi'] = phi
		
		df['event id'] = event_id_head + df['event id'].astype(str)
		
		df = df[['event id', 'Particle', 'Energy', 'theta', 'phi', 'p_spot', 'p_track', 'p_worm', 'length', 'width']]
		
		data = data.append(df, ignore_index=True)

data.to_csv('./mc-truth.csv', index=False)
		
		
		
	


