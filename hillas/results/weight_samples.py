import pandas as pd
import numpy as np
import csv
import os

# we have 10 discrete theta values (degrees):
# 0.0
# 25.8
# 36.9
# 45.6
# 53.1
# 60.0
# 66.4
# 72.5
# 78.5
# 84.3

# atmospheric muon flux goes with cos^2(theta)
# the effective area goes, roughly, with cos(theta)

# each weighted by (cos(theta))^3

# we create a csv with our new refined sample
outfile = './sample2.csv'
#muons_dir = './simulation-test/simulation/sample_stats/mu+/'
muons_dir = './simulation-threshold/sample_stats/mu+/'

# each run has 100 events
# we take the first (100 * cos(theta)^3) events from each run and add them to the sample

sample = pd.DataFrame()
for file in os.listdir(muons_dir):
	path = muons_dir + file
	
	theta = float( file.split('_')[2] ) * np.pi / 180.0
	
	energy_scale = file.split('_')[0][-3:]
	
	weight = int(np.ceil((np.cos(theta) ** 3 * 90)))
	
	df = pd.read_csv(path)
	
	if(energy_scale == "GeV"):
		sample = sample.append(df.head(weight), ignore_index=True)
	
		#print(len(df.index))
	
sample.to_csv(outfile)