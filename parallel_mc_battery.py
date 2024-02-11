# flake8: noqa
import time
import logging
import numpy as np

import apache_beam as beam

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Parallel_MC_Battery:
    '''
    Helper class to orchestrate in parallel Monte Carlo simulations for an arbitrary number of models, 
    with low-level parameter granularity.
    '''
    def __init__(self, battery_configs, pipeline_options):
        self.battery_configs = battery_configs
        self.pipeline_options = pipeline_options
        
    def simulate(self, models, simulation_configs, output_paths="."):
        class SimulateDoFn(beam.DoFn):
            bundle_counter = 0

            def start_bundle(self):
                logging.info(f"Bundle {type(self).bundle_counter} created, and sent to workers for parallel simulation...")
                type(self).bundle_counter += 1

            def process(self, element):
                model, simulation_configs, output_path= element

                parameters = simulation_configs["parameters"]
                starting_point = simulation_configs["starting_point"]
                number_points = simulation_configs["number_points"]
                number_simulations = simulation_configs["number_simulations"]

                monte_carlo_point = starting_point
                fixed_model = model(parameters)

                monte_carlo_traces = []
                monte_carlo_trace = []
                monte_carlo_trace.append(monte_carlo_point)

                for _ in range(number_simulations):
                    for _ in range(number_points):
                        monte_carlo_point = fixed_model(monte_carlo_point)
                        monte_carlo_trace.append(monte_carlo_point)
                    
                    monte_carlo_traces.append(monte_carlo_trace)
                    monte_carlo_point = starting_point
                
                yield (monte_carlo_traces, output_path)
        


