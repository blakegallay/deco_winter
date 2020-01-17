import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Plotting distributions of Hillas parameters
# from simulation and real DECO images

# Samples:
# 	Simulation - All mu+ runs
#	DECO - all 'Track' events

# Parameters to plot:
#	length
#	width

columns = ['id', 'p_spot', 'p_track', 'p_worm', 'x_coord', 'y_coord', 'width', 'length', 'azwidth']

tracks = pd.read_csv('./deco/Track.csv')
spots = pd.read_csv('./deco/Spot.csv')

muons_dir = './simulation-threshold/sample_stats/mu+/'
e_dir = './simulation-threshold/sample_stats/e+/'
#muons_dir = './simulation/sample_stats/mu+/'
#e_dir = './simulation/sample_stats/e+/'

muons = pd.DataFrame()
'''
for file in os.listdir(muons_dir):
	path = muons_dir + file
	
	muons = muons.append(pd.read_csv(path), ignore_index=True)
'''

muons = pd.read_csv('./sample.csv')

electrons = pd.DataFrame()
for file in os.listdir(e_dir):
	path = e_dir + file
	
	electrons = electrons.append(pd.read_csv(path), ignore_index=True)
	
	
#muons['p_spot'] = muons['p_spot'].astype(float)
#print(muons['p_spot'])
#print(muons)

# Exclude spots
#muons = muons[(muons['p_spot'] < 0.3)]
#muons = muons[(muons['p_track'] > 0.95)]
#muons = muons[(muons['p_spot'] < 0.07)]
#print(muons)
	
lengths = {'tracks': tracks['length'].tolist(), 'spots': spots['length'].tolist(), 'mu+': muons['length'].tolist(), 'e+': electrons['length'].tolist()}
widths = {'tracks': tracks['width'].tolist(), 'spots': spots['width'].tolist(), 'mu+': muons['width'].tolist(), 'e+': electrons['width'].tolist()}

#bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40]
#bins = [0.1 + 0.01 * n for n in range(90)] + [1 + 0.1 * n for n in range(100)]
#bins = [0] + [0.05 * n for n in range(20)] + [1 + 0.5 * n for n in range(18)] + [10 + 5 * n for n in range(19)]

bins = np.logspace(-1, 1, 100)

plt.xscale('log')
plt.ylabel("Probability Density (1/pixels)")
plt.xlabel("WIDTH (pixels, ~um)")
plt.title("Distribution of WIDTH")

plt.hist(widths['spots'], bins=bins, alpha=0.5, label='DECO Spots', density=True)
plt.hist(widths['tracks'], bins=bins, alpha=0.5, label='DECO Tracks', density=True)
plt.hist(widths['mu+'], bins=bins, alpha=0.5, label='Simulated mu+', density=True)
plt.legend(loc='upper right')
	
plt.savefig('./width_dist.png')
	
plt.close()



#bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40]

#bins = [0.1 + 0.01 * n for n in range(90)] + [1 + 0.1 * n for n in range(100)]

#bins = [0] + [0.05 * n for n in range(20)] + [1 + 0.5 * n for n in range(18)] + [10 + 5 * n for n in range(19)]
bins = np.logspace(-0.25, 1.5, 100)

plt.xscale('log')
plt.ylabel("Probability Density (1/pixels)")
plt.xlabel("LENGTH (pixels, ~um)")
plt.title("Distribution of LENGTH")

plt.hist(lengths['spots'], bins=bins, alpha=0.5, label='DECO Spots', density=True)
plt.hist(lengths['tracks'], bins=bins, alpha=0.5, label='DECO Tracks', density=True)
plt.hist(lengths['mu+'], bins=bins, alpha=0.5, label='Simulated mu+', density=True)
#plt.hist(lengths['e+'], bins=bins, alpha=0.5, label='Simulated e+', density=True)
plt.legend(loc='upper right')
	
plt.savefig('./length_dist.png')
	
plt.close()

print("Width medians:")
print("DECO: " + str(np.median(widths['tracks'])) )
print("Simulation: " + str(np.median(widths['mu+'])) )