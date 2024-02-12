# flake8: noqa
import time
import logging

import numpy as np
from pydantic import BaseModel, validator, ValidationError

import apache_beam as beam


class BatteryConfigs(BaseModel):
    rng: str
    mode: str

    @validator("battery_configs")
    def values_allowed_options(cls, configs):
        allowed_rngs = ["PCG64", "Philox", "SFC64", "MT19937"]
        allowed_modes = ["production", "testing"]

        if cls.rng not in allowed_rngs:
            raise ValueError(
                f"Unsupported RNG choice. Allowed options: {' ,'.join(allowed_rngs)}"
            )

        if cls.mode not in allowed_modes:
            raise ValueError(
                f"Unsupported mode choice. Allowed options: {' ,'.join(allowed_modes)}"
            )

        return configs


class SimulationConfigs(BaseModel):
    parameters: list[float]

    simulation_configs: dict[list[float], float, int, int]

    @validator("simulation_configs")
    def values_math_relevant(cls, configs):
        if configs["number_points"] < 1:
            raise ValueError(
                "The minimum number of points in a single Monte Carlo\
                                              trace should be >= 1."
            )

        if configs["number_simulations"] < 1:
            raise ValueError("The minimum number of Monte Carlo traces should be >= 1.")

        return configs


class ParallelMCBattery:
    """
    Helper class to orchestrate in parallel Monte Carlo simulations for an arbitrary number of models,
    with low-level parameter granularity.
    """

    battery_configs = {"rng": "PCG64", "mode": "testing"}

    def __init__(self, pipeline_options, battery_configs=None):
        battery_configs = battery_configs or type(self).battery_configs

        try:
            battery_configs = BatteryConfigs(battery_configs)
            type(self).mode = battery_configs["mode"]
            rng_mapping = {
                    "PCG64": np.random.PCG64,
                    "Philox": np.random.Philox,
                    "SFC64": np.random.SFC64,
                    "MT19937": np.random.MT19937,
                }
            
            rng_generator = rng_mapping[battery_configs["rng"]]

            self.rng = np.random.default_rng(rng_generator())
            self.rng_64_bits = rng_generator()
        except KeyError:
            logging.exception("Missing parameters from battery_configs")
        except ValidationError:
            logging.exception("Validation of battery_configs failed")
            raise

        type(self).pipeline_options = pipeline_options

    def simulate(self, models, simulation_configs, output_paths="."):
        class SimulateDoFn(beam.DoFn):
            def start_bundle(self):
                logging.info(
                    f"New bundle created,\
                              and sent to workers for parallel simulation..."
                )

            def process(self, element):
                model, simulation_configs, output_path = element

                try:
                    simulation_configs = SimulationConfigs(simulation_configs)
                except KeyError:
                    logging.exception("Missing parameters from simulation_configs")
                except ValidationError:
                    logging.exception("Validation of simulation_configs failed")
                    raise

                number_points = simulation_configs["number_points"]
                number_simulations = simulation_configs["number_simulations"]

                monte_carlo_traces = []
                monte_carlo_trace = []

                for _ in range(number_simulations):
                    if self.mode == "production":
                        parameters = simulation_configs["parameters"]
                        fixed_model = model(parameters, self.rng)
                        starting_point = simulation_configs["starting_point"]
                        monte_carlo_point = starting_point

                        for _ in range(number_points):
                            monte_carlo_point = fixed_model(monte_carlo_point)
                            monte_carlo_trace.append(monte_carlo_point)
                    else:
                        for _ in range(number_points):
                            monte_carlo_point = format(
                                self.rng_64_bits.random_raw(), "064b"
                            )
                            monte_carlo_trace.append(monte_carlo_point)

                    monte_carlo_traces.append(monte_carlo_trace)

                yield (monte_carlo_traces, output_path)
