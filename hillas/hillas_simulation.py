import numpy as np
import os
import csv
from scipy.misc import imread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zoom_predictions_4class_modified import *
import time

cnn_weights_file = 'final_trained_weights.h5'
cnn_model = build_model(4)
cnn_model.load_weights(cnn_weights_file)

# from Brent, calculate Hillas parameters of events
#charge_coords = [[x_coords], [y_coords], [charges]]
def hillas(charge_coords):
        #print(charge_coords.shape)
        x = 0
        y = 0
        x2 = 0
        y2 = 0
        xy = 0
        CHARGE = 0
        #print(charge_coords.shape)
        CHARGE = np.sum(charge_coords[2])
        x = np.sum(charge_coords[0] * charge_coords[2])
        y = np.sum(charge_coords[1] * charge_coords[2])
        x2 = np.sum(charge_coords[0] ** 2 * charge_coords[2])
        y2 = np.sum(charge_coords[1] ** 2 * charge_coords[2])
        xy = np.sum(charge_coords[0] * charge_coords[1] * charge_coords[2])
        x /= CHARGE
        y /= CHARGE
        x2 /= CHARGE
        y2 /= CHARGE
        xy /= CHARGE
        S2_x = x2 - x ** 2
        S2_y = y2 - y ** 2
        S_xy = xy - x * y
        d = S2_y - S2_x
        a = (d + np.sqrt(d ** 2 + 4 * S_xy ** 2)) / (2 * S_xy)
        b = y - a * x
        width = np.sqrt((S2_y + a ** 2 * S2_x - 2 * a * S_xy) / (1 + a ** 2))
        length = np.sqrt((S2_x + a ** 2 * S2_y + 2 * a * S_xy) / (1 + a ** 2))
        miss = np.abs(b / np.sqrt(1 + a ** 2))
        dis = np.sqrt(x ** 2 + y ** 2)
        q_coord = (x - charge_coords[0]) * (x / dis) + (y - charge_coords[1]) * (y / dis)
        q = np.sum(q_coord * charge_coords[2]) / CHARGE
        q2 = np.sum(q_coord ** 2 * charge_coords[2]) / CHARGE
        azwidth = q2 - q ** 2
        return [width, length, miss, dis, azwidth]
		
# SIMULATION
# We read the raw .txt data from the simulation
# using the coordinates reported by the CNN as the origin

# deprected, used for testing
#data_path = "../sim-txts/1GeV_theta_25.8_phi_50.0_highstats.txt"
#cnn_output_path = "../classifications/1GeV_theta_25.8_phi_50.0_highstats.csv"

source_dir = "../simulation_images/"
export_dir = "./results/simulation/sample_stats/"

'''
with open(cnn_output_path, "r") as cnnfile:
	reader = csv.reader(cnnfile)
	data = list(reader)
	
	# Origin coordinates
	x = int(data[1][7])
	y = int(data[1][8])'''
	
# same reading code as sim-jpgs.py
#txt = open(data_path, "r").read().split("\n")

# master stats csvs, contains stats on CNN probabilities and Hillas parameters for each simulation run/sample
# e+.csv, gamma.csv, mu+.csv
stats_dir = './results/simulation/cumulative_stats/'
master_data = {}

# background noise array

#background noise parameters						
sigma = 0.1
mu = 0.2
# simulation image dimensions
width = 2592
height = 1944
background_array = np.random.randn(height, width) * sigma + mu

for file in os.listdir(stats_dir):
	with open(stats_dir + file, 'r') as statsfile:
		reader = csv.reader(statsfile)
		category = file[:-4]
		master_data[category] = {}
		for row in reader:
			# row[0] = image_name, i.e. '1GeV_theta_0.0_phi_0.0_highstats'
			if(len(row) > 0):
				master_data[category][row[0]] = row[1:]
	
for category in os.listdir(source_dir): # e+, gamma, mu+

	for sample in os.listdir(source_dir + category):
	
		arrs = {}
	
		widths = []
		lengths = []
		azwidths = []
	
	
		sample_name = sample[:-4] # remove .txt file extension
		if(sample_name not in master_data[category]):
			start_time = time.time()
			# new image
			
			# read txt 
			line_num = 0
			hillas_params = {}
			
			txt = open(source_dir + category + "/" + sample, "r").read().split("\n")
			
			num_events = 0
			for line in txt:
				line_num += 1
				if "===" in line:
					num_events += 1
						
					data = []
						
					for entry in txt[line_num:]:
						if "===" in entry:
							break
						elif "---" in entry or "#" in entry or len(entry) == 0:
							continue
						else:
							pixel_data = entry[9:].split(", ")
							data.append(pixel_data)
					
					charge_coords = []

					charge_x = []
					charge_y = []
					charges = []
					
					for d in data:
						charge_x.append(int(d[0]))
						charge_y.append(int(d[1]))
						charges.append(float(d[2]))
						
					event_hillas_params = hillas([np.asarray(charge_x), np.asarray(charge_y), np.asarray(charges)])
					
					invalid = False
					
					for num in event_hillas_params:
						if np.isnan(num):
							invalid = True
				
					if(not invalid):
						
						widths.append(event_hillas_params[0])
						lengths.append(event_hillas_params[1])
						azwidths.append(event_hillas_params[4])
						
						hillas_params[num_events] = event_hillas_params
						
						array = np.zeros([height, width])
						for n in range(50):
							for m in range(50):
								x = int(data[0][0]) + 25 - n
								y = int(data[0][1]) + 25 - m
								try:
									array[y, x] = background_array[y, x]
								except IndexError:
									continue
						
						for d in data:
							array[int(d[1]),int(d[0])] = float(d[2])
						
						arrs[num_events] = array
						if(len(arrs)) == 1:
							first_event_num = num_events
							
			elapsed_time = time.time() - start_time
			print("HILLAS TIME: " + str(elapsed_time))
				
			# write image parameters to dedicated csv
			with open(export_dir + category + "/" + sample_name + ".csv", 'w') as out:
				writer = csv.writer(out)
				headings = ["event id", "p_spot", "p_track", "p_worm", "x_coord", "y_coord", "width", "length", "azwidth"]
				
				writer.writerow(headings)
				
				if(len(arrs) > 0):
				
					# run images through CNN
					
					# until runtime efficiency fix is found, we only run the first event from each sample through the CNN
					cnn_data = {}
					
					
					start_time = time.time()
					df = run_blob_classifier(arrs, 4, cnn_model)
					elapsed_time = time.time() - start_time
					print("CNN TIME: " + str(elapsed_time))
					
					#print(df)
					
					for i in range(len(df.index)):
						cnn_data[df['event_id'][i]] = [df['p_noise'][i], df['p_spot'][i], df['p_track'][i], df['p_worm'][i], df['x_coord'][i], df['y_coord'][i]]
					
					for i in hillas_params:
					
						p = hillas_params[i]
						
						c = cnn_data[i]
						
						'''
						if(i == first_event_num):
							c = cnn_data[1]
						else:
							c = ["", "", "", "", "", ""]
						'''
						
						row = [i, c[1], c[2], c[3], c[4], c[5], p[0], p[1], p[4]]

						writer.writerow(row)
				
			# append row to master stats csv
			if(len(arrs) > 0):
				master_data[category][sample_name] = [df['p_spot'].mean(), df['p_track'].mean(), df['p_worm'].mean(), np.mean(widths), np.mean(lengths), np.mean(azwidths), df['x_coord'].mean(), df['y_coord'].mean()]
				
				with open(stats_dir + category + ".csv", 'w') as statsfile:
					writer = csv.writer(statsfile)
					headings = ["sample", "p_spot", "p_track", "p_worm", "width", "length", "azwidth", "x_coord", "y_coord"]
					writer.writerow(headings)
					for d in master_data[category]:
						writer.writerow([d] + master_data[category][d])
					