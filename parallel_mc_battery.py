# flake8: noqa
import time
import logging

import numpy as np
from pydantic import BaseModel, validator, ValidationError

import apache_beam as beam


class BatteryConfigs(BaseModel):
    rng: str
    mode: str

    @validator("rng")
    def validate_rng(cls, rng):
        allowed_rngs = ["PCG64", "Philox", "SFC64", "MT19937"]

        if rng not in allowed_rngs:
            raise ValueError(
                f"Unsupported RNG choice. Allowed options: {' ,'.join(allowed_rngs)}"
            )

        return rng

    @validator("mode")
    def validate_mode(cls, mode):
        allowed_modes = ["production", "testing"]

        if mode not in allowed_modes:
            raise ValueError(
                f"Unsupported mode choice. Allowed options: {' ,'.join(allowed_modes)}"
            )

        return mode


class SimulationConfigs(BaseModel):
    parameters: list[float]
    starting_point: float
    number_simulations: int
    number_points: int

    @validator("number_simulations")
    def validate_number_simulations(cls, number_simulations):
        if number_simulations < 1:
            raise ValueError(
                "The minimum number of Monte Carlo simulations\
                                               should be >= 1."
            )

        return number_simulations

    @validator("number_points")
    def validate_number_points(cls, number_points):
        if number_points < 1:
            raise ValueError(
                "The minimum number of points in a single Monte Carlo\
                                              trace should be >= 1."
            )

        return number_points


class ParallelMCBattery:
    """
    Helper class to orchestrate in parallel Monte Carlo simulations for an arbitrary number of models,
    with low-level parameter granularity.
    """

    battery_configs = {"rng": "PCG64", "mode": "testing"}

    def __init__(self, pipeline_options, battery_configs=None):
        try:
            battery_configs = battery_configs or type(self).battery_configs
            rng = battery_configs["rng"]
            mode = battery_configs["mode"]

            battery_configs = BatteryConfigs(rng=rng, mode=mode)
            type(self).mode = battery_configs.mode
            rng_mapping = {
                "PCG64": np.random.PCG64,
                "Philox": np.random.Philox,
                "SFC64": np.random.SFC64,
                "MT19937": np.random.MT19937,
            }

            rng_generator = rng_mapping[battery_configs.rng]

            self.rng = np.random.default_rng(rng_generator())
            self.rng_64_bits = rng_generator()
        except KeyError:
            logging.exception("Missing parameters from battery configuration")
            raise
        except ValidationError:
            logging.exception("Validation of battery configuration failed")
            raise

        type(self).pipeline_options = pipeline_options

    def simulate(self, models, simulation_configs, output_paths="."):
        try:
            for simulation_config in simulation_configs:
                parameters = simulation_config["parameters"]
                starting_point = simulation_config["starting_point"]
                number_simulations = simulation_config["number_simulations"]
                number_points = simulation_config["number_points"]
                simulation_config = SimulationConfigs(
                    parameters=parameters,
                    starting_point=starting_point,
                    number_simulations=number_simulations,
                    number_points=number_points,
                )
        except KeyError:
            logging.exception("Missing parameters from simulation configuration")
            raise
        except ValidationError:
            logging.exception("Validation of simulation configurations failed")
            raise

        class SimulateDoFn(beam.DoFn):
            def start_bundle(self):
                logging.info(
                    f"New bundle created,\
                              and sent to workers for parallel simulation..."
                )

            def process(self, element):
                model, simulation_configs, output_path = element

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
