# flake8: noqa
import time
import numpy as np

import apache_beam as beam

class Parallel_MC_Battery:
    '''
    Helper class to orchestrate in parallel Monte Carlo simulations for an arbitrary number of models, 
    with low-level parameter granularity.
    '''
    def __init__(self, battery_configs, pipeline_options):
        self.battery_configs = battery_configs
        self.pipeline_options = pipeline_options
        
    def simulate(self, models, parameters, starting_points, number_simulations, output_paths):
        class SimulateDoFn(beam.DoFn):
            def process(self, element):
                model, individual_parameters, starting_point, number_individual_simulations, output_path = element

                monte_carlo_point = starting_point
                fixed_model = model(individual_parameters)

                monte_carlo_trace = [].append(monte_carlo_point)
                for _ in number_individual_simulations:
                    monte_carlo_point = fixed_model(monte_carlo_point)
                    monte_carlo_trace.append()
                
                yield monte_carlo_trace
        
        

