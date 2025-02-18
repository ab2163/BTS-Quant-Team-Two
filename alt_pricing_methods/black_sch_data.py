from py_vollib.black_scholes import black_scholes
from random import uniform
import numpy as np
import torch
from torch.utils.data import Dataset

# Function which generates artificial dataset from Blacks Scholes equation
def generate_black_sch_data():
	# Option parameters
	NUM_SAMPLES = 35000
	TEST_RATIO = 1/7
	SPOT_UPP = 500
	SPOT_LOW = 10
	STR_UPP = 1.1
	STR_LOW = 0.6
	MAT_UPP = 2.0
	MAT_LOW = 0.01
	VOL_UPP = 0.2
	VOL_LOW = 0.1
	INT_UPP = 0.20
	INT_LOW = 0.05

	# Randomly generate values of spot price, strike price etc.
	spots = np.array([uniform(SPOT_LOW, SPOT_UPP) for p in range(0, NUM_SAMPLES)])
	strikes = np.array([uniform(STR_LOW*spots[p], STR_UPP*spots[p]) for p in range(0, NUM_SAMPLES)])
	mat_times = np.array([uniform(MAT_LOW, MAT_UPP) for p in range(0, NUM_SAMPLES)])
	vols = np.array([uniform(VOL_LOW, VOL_UPP) for p in range(0, NUM_SAMPLES)])
	int_rates = np.array([uniform(INT_LOW, INT_UPP) for p in range(0, NUM_SAMPLES)])

	# Calculate Black-Scholes option prices
	call_prices = []
	for i in range(0, NUM_SAMPLES):
	    call_prices.append(black_scholes('c', spots[i], strikes[i], mat_times[i], int_rates[i], vols[i]))

	# Calculate means
	spots_mean = spots.mean()
	strikes_mean = strikes.mean()
	mat_times_mean = mat_times.mean()
	vols_mean = vols.mean()
	int_rates_mean = int_rates.mean()

	# Calculate standard deviations
	spots_std = spots.std()
	strikes_std = strikes.std()
	mat_times_std = mat_times.std()
	vols_std = vols.std()
	int_rates_std = int_rates.std()

	# Normalize data
	spots = (spots - spots_mean)/spots_std
	strikes = (strikes - strikes_mean)/strikes_std
	mat_times = (mat_times - mat_times_mean)/mat_times_std
	vols = (vols - vols_mean)/vols_std
	int_rates = (int_rates - int_rates_mean)/int_rates_std

	# Return data
	bl_sch_data = {'spots': spots, 
					'strikes': strikes, 
					'mat_times': mat_times,
					'vols': vols, 
					'int_rates': int_rates,
					'call_prices': call_prices,
					'NUM_SAMPLES': NUM_SAMPLES,
					'TEST_RATIO': TEST_RATIO}
	return bl_sch_data

# Defines a dataset using aritificial data used by training algorithm
class BlackSchDataset(Dataset):
	def __init__(self, training_set):

		# Generate the Black Scholes data
		self.gen_data = generate_black_sch_data()
		self.spots = self.gen_data['spots']
		self.strikes = self.gen_data['strikes']
		self.mat_times = self.gen_data['mat_times']
		self.vols = self.gen_data['vols']
		self.int_rates = self.gen_data['int_rates']
		self.call_prices = self.gen_data['call_prices']
		self.NUM_SAMPLES = self.gen_data['NUM_SAMPLES']
		self.TEST_RATIO = self.gen_data['TEST_RATIO']

		if training_set:
			self.len_data = int((1 - self.TEST_RATIO)*self.NUM_SAMPLES)
			self.offset = 0
		else:
			self.len_data = int(self.TEST_RATIO*self.NUM_SAMPLES)
			self.offset = int((1 - self.TEST_RATIO)*self.NUM_SAMPLES)

	def __len__(self):
		return self.len_data

	def __getitem__(self, idx):
		idx = idx + self.offset
		input_vals = torch.tensor([self.spots[idx], 
									self.strikes[idx], 
									self.mat_times[idx], 
									self.int_rates[idx], 
									self.vols[idx]])
		label = torch.tensor([self.call_prices[idx]])
		return input_vals, label