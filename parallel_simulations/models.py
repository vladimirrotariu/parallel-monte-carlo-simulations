import os
import logging
from typing import Optional, Union, List

from apache_beam.options.pipeline_options import (
    PipelineOptions,
)

from pydantic import BaseModel, validator, ValidationError


class BatteryConfigs(BaseModel):
    rng: Optional[str] = None
    pipeline_options: PipelineOptions

    @validator("rng")
    def validate_rng(cls, rng: str) -> str:
        allowed_rngs = [
            "PCG64",
            "Philox",
            "SFC64",
            "MT19937",
        ]

        if rng is None:
            rng = "PCG64"

        if rng not in allowed_rngs:
            raise ValueError(
                f"Unsupported RNG choice. Allowed options: {' ,'.join(allowed_rngs)}"
            )

        return rng

    class Config:
        arbitrary_types_allowed = True


class SimulationConfigs(BaseModel):
    number_simulations: int
    number_points: int
    parameters: Optional[Union[int, float, List[float], List[int]]] = None
    starting_point: Optional[Union[float, str, List[float], List[str]]] = None

    @validator("number_simulations")
    def validate_number_simulations(cls, number_simulations: int) -> int:
        if number_simulations < 1:
            raise ValueError(
                "The minimum number of Monte Carlo simulations should be >= 1."
            )

        return number_simulations

    @validator("number_points")
    def validate_number_points(cls, number_points: int) -> int:
        if number_points < 1:
            raise ValueError(
                "The minimum number of points in a single Monte Carlo trace should be >= 1."
            )

        return number_points


class OutputPath(BaseModel):
    output_path: str

    @validator("output_path")
    def validate_output_path(cls, output_path: str) -> str:
        directory_path, file_name = os.path.split(output_path)

        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            logging.info(f"Directory {directory_path} created.")

        if file_name and not os.path.exists(output_path):
            try:
                with open(output_path, "x") as f:
                    f.write("")
                logging.info(f"File {output_path} created.")
            except PermissionError:
                raise PermissionError(
                    f"Not enough permissions to write to the {file_name} file"
                )

        return output_path
