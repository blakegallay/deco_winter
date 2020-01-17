import numpy as np
import os
import csv
from imageio import imread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zoom_predictions_4class import run_blob_classifier

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
		
# DECO IMAGES
# ONLY TRACKS

# Read CNN output csv

# mean stdev median 1st quartile 3rd quartile min max
stats = {}

headings = ["event_id", "p_noise", "p_spot", "p_track", "p_worm", "x_coord", "y_coord"]

export_dir = './results/deco'

paths = []
paths_file = "./Spot_imagepaths.csv"
with open(paths_file, 'r') as p:
	reader = csv.reader(p)
	n = 0
	for row in reader:
		n += 1
		#if(n < 10):
		if(True):
			paths.append(row[0])

cnn_output = {}

cnn_weights_file = 'final_trained_weights.h5'
print('classifying')
df = run_blob_classifier(paths, 'out.csv', 4, weights_file=cnn_weights_file)
print('done classifying')
hillas_params = []

widths = []
lengths = []
azwidths = []
for path in paths:
		
	cnn_row = df.loc[df['image_file'] == path]
	if(cnn_row.empty):
		continue
	index = cnn_row.index.values[0]
	cnn_data = [cnn_row['p_spot'][index], cnn_row['p_track'][index], cnn_row['p_worm'][index], cnn_row['x_coord'][index], cnn_row['y_coord'][index]]

	origin = [int(cnn_data[3]), int(cnn_data[4])]

	image = imread(path, as_gray=True)
		
	charge_x = []
	charge_y = []
	charges = []
	
	# We run the CNN on each image. We save the CNN output along with the Hillas parameters for each image
	
	# We get the 'center'/origin coordinate from the CNN output 
		
	event_id = path.split("/")[-1][:-4]
		
	radius_threshold = 5 # pixels, ~1um/pixel
	luminosity_threshold = 50 # unitless, ~deposited charge
		
	brights = np.argwhere(image > luminosity_threshold)

	for n in range(brights.shape[0]):
		x = brights[n,0]
		y = brights[n,1]
			
		if((y - origin[0]) < radius_threshold and (x - origin[1]) < radius_threshold): 
			charge_x.append(x-origin[1])
			charge_y.append(y-origin[0])
			charges.append(image[x,y])
			
	event_hillas_params = cnn_data[0:3] + hillas([np.asarray(charge_x), np.asarray(charge_y), np.asarray(charges)]) + cnn_data[3:]
		
	invalid = False
	for num in event_hillas_params:
		if np.isnan(num):
			invalid = True
				
	if(not invalid):
		hillas_params.append([event_id] + event_hillas_params)
		widths.append(event_hillas_params[3])
		lengths.append(event_hillas_params[4])
		azwidths.append(event_hillas_params[5])

with open('./results/Spot.csv', 'w') as outfile:
	writer = csv.writer(outfile)
	writer.writerow(["event_id", "p_spot", "p_track", "p_worm", "width", "length", "azwidth", "x_coord", "y_coord"])
	for p in hillas_params:
		writer.writerow(p)

with open('./results/Spotstats.csv', 'w') as statsfile:
	writer = csv.writer(statsfile)
	writer.writerow(['variable', 'mean', 'stdev', 'min', 'Q1', 'median', 'Q3', 'max'])
	variables = {"width":widths, "length":lengths, "azwidth":azwidths}
	for name in variables:
		var = variables[name]
		writer.writerow([name, np.mean(var), np.std(var), np.min(var)] + list(np.percentile(var, [25,50,75])) + [np.max(var)])	
# make stats summary for each variable
'''
for name in variables:
	var = variables[name]
	stats[dir][name] = [np.mean(var), np.std(var)] + [np.min(var)] + list(np.percentile(var, [25,50,75])) + [np.max(var)]
'''
