import json
import random
import string
import socket
import os
import glob
import pdb
from datetime import datetime

import numpy as np
import gym
import pickle

def get_run_name(args):
	current_date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
	return str(current_date)+"_"+str(args.env)+"_"+str(args.algo)+"_t"+str(args.timesteps)+"_seed"+str(args.seed)+"_"+socket.gethostname()

def get_random_string(n=5):
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

def set_seed(seed):
	if seed > 0:
		np.random.seed(seed)

def create_dir(path):
	try:
		os.mkdir(os.path.join(path))
	except OSError as error:
		# print('Dir already exists')
		pass

def create_dirs(path):
	try:
		os.makedirs(os.path.join(path))
	except OSError as error:
		pass

def save_config(config, path):
    with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as file:
        # pprint(vars(config), stream=file)
        json.dump(config, file)
    return

def load_config(path):
	with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8') as file:
		config = json.load(file)
	return config

def get_learning_rate(args, env):
	"""
		Priority:
			args.lr > env.preferred_lr > 0.0003 (default)
	"""
	if args.lr is None:
		if env.get_attr('preferred_lr')[0] is None:
			return 0.0003
		else:
			return env.get_attr('preferred_lr')[0]
	else:
		return args.lr

def load_object(filepath):
    return pickle.load(open(filepath, 'rb'))

def save_object(obj, save_dir, filename):
    with open(os.path.join(save_dir, f'{filename}.pkl'), 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)