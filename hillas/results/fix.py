import csv
import numpy as np
import pandas as pd
import os

test_dir = './simulation-threshold/sample_stats/gamma/'
orig_dir = './simulation-original/sample_stats/gamma/'

for path in os.listdir(test_dir):
	test_path = test_dir + path
	#orig_path = orig_dir + path
	
	test_data = pd.read_csv(test_path)
	
	try:
		test_data.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
		test_data.drop(["a"], axis=1, inplace=True)
	except KeyError:
		pass
	
	try:
		test_data.rename({"Unnamed: 0.1":"b"}, axis="columns", inplace=True)
		test_data.drop(["b"], axis=1, inplace=True)
	except KeyError:
		pass
	
	try:
		test_data.rename({"Unnamed: 0.1.1":"c"}, axis="columns", inplace=True)
		test_data.drop(["c"], axis=1, inplace=True)
	except KeyError:
		pass
	
	#print(test_data)
	
	#orig_data = pd.read_csv(orig_path)
	
	try:
		pass
		#test_data = test_data.join(orig_data['p_spot'])
		#test_data = test_data.join(orig_data['p_track'])
		#test_data = test_data.join(orig_data['p_worm'])
	except ValueError:
		print('error')
		continue
	#print(test_data)
	
	test_data.to_csv(test_path)
