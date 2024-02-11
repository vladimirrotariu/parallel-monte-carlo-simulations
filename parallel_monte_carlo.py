# flake8: noqa
import time
import numpy as np

import apache_beam as beam

# model_conf with  e.g. rng="PCG64", mode="production", seeds=None

class Parallel_Monte_Carlo:
    def __init__(self, parameters, number_simulations, model_configs, output_path):
        self.parameters = parameters
        self.number_simulations = number_simulations
        self.model_configs = model_configs
        self.output_path = output_path
        
    
