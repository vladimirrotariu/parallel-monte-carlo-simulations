# flake8: noqa
import time
import logging
import numpy as np
from pydantic import BaseModel, validator

import apache_beam as beam

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ParallelMCBattery:
    '''
    Helper class to orchestrate in parallel Monte Carlo simulations for an arbitrary number of models, 
    with low-level parameter granularity.
    '''
    def __init__(self, battery_configs, pipeline_options):
        
        class BatteryConfigs(BaseModel):
            battery_configs: dict[str, str] #rng, mode (production, testing)

            @validator("battery_configs")
            def values_allowed_options(cls, configs):
                allowed_rngs = ["PCG64", "Philox", "SFC64", "MT19937"]
                allowed_modes = ["production", "testing"]

                if configs["rng"] not in allowed_rngs:
                    raise ValueError(f"The chosen RNG could not be found.\
                                        Allowed options: {str(allowed_rngs)[1,-1]}")
                
                if configs["mode"] not in allowed_modes:
                    raise ValueError(f"The chosen mode could not be found.\
                                        Allowed options: {str(allowed_modes)[1,-1]}")
                

        try:        
            battery_configs = BatteryConfigs(battery_configs)
        except ValueError:
            logging.exception("Validation of battery_configs failed")

        self.battery_configs = battery_configs
        self.pipeline_options = pipeline_options
        
    def simulate(self, models, simulation_configs, output_paths="."):
        class SimulateDoFn(beam.DoFn):
            bundle_counter = 0

            def start_bundle(self):
                logging.info(f"Bundle {type(self).bundle_counter} created,\
                              and sent to workers for parallel simulation...")
                type(self).bundle_counter += 1

            def process(self, element):
                model, simulation_configs, output_path= element

                try:
                    simulation_configs = self.__validate_simulation_configs(simulation_configs)
                except ValueError:
                    logging.exception("Validation of simulation_configs failed")

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
        
            def __validate_simulation_configs(simulation_configs):
                class SimulationConfigs(BaseModel):
                    simulation_configs: dict[list[float], float, int, int]
                    
                    @validator("simulation_configs")
                    def values_math_relevant(cls, configs):
                        if configs["number_points"] < 1:
                            raise ValueError("The minimum number of points in a single Monte Carlo\
                                              trace should be >= 1.")
                        
                        if configs["number_simulations"] < 1:
                            raise ValueError("The minimum number of Monte Carlo traces should be >= 1.")
                        return configs

                return SimulationConfigs(simulation_configs)

        
