import os
import scipy.misc
import copy
import numpy as np
# image dimensions: 2592 x 1944 hmm
width = 2592
height = 1944
# pixel dimensions: 0.9um x 0.9um
# sensor thickness: 25.0um

# Converts a txt file describing simulated DECO events into an RGB jpg image

source_dir = "./sim-txts"
export_dir = "./sim-jpgs"

files = os.listdir(source_dir)

for path in files:
	
	arrs = []

	#blank_array = [[0 for p in range(width)] for p in range(height)]
	
	txt = open(source_dir + "/" + path, "r").read().split("\n")
	
	line_num = 0
	for line in txt:
		line_num += 1
		if "===" in line:
			
			data = []
			
			for entry in txt[line_num:]:
				if "===" in entry:
					break
				elif "---" in entry or "#" in entry or len(entry) == 0:
					continue
				else:
					pixel_data = entry[9:].split(", ")
					data.append(pixel_data)
					
			array = np.zeros([height, width])
			for d in data:
				array[int(d[1]),int(d[0])] = float(d[2])
#				print(array[int(d[1]), int(d[0])])
			arrs.append(array)
		
	n = 0
	if(not os.path.isdir(export_dir+"/"+path[:-4])):
		os.mkdir(export_dir+"/"+path[:-4])
		for arr in arrs:
			n += 1
	#		print(arr)
			scipy.misc.imsave(export_dir+"/"+path[:-4]+"/"+str(n)+".jpg", arr)
		

			

