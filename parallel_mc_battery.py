# flake8: noqa
import time
import numpy as np

import apache_beam as beam

class Parallel_MC_Battery:
    '''
    Helper class to orchestrate in parallel Monte Carlo simulations for an arbitrary number of models.
    '''
    def __init__(self, battery_configs, pipeline_options):
        self.battery_configs = battery_configs
        self.pipeline_options = pipeline_options
        
    def simulate(self, model, parameters, number_simulations, output_path):
        self.model = model
        self.parameters = parameters
        self.number_simulations = number_simulations
        self.output_path = output_path


