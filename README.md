# Parallelize statistical simulations with Apache Beam
## ParallelMCBattery class to parallelize Monte Carlo simulations
### Installation
```bash
pip install "git+https://github.com/vladimirrotariu/parallel-monte-carlo-simulations#egg=parallel_mc_battery&subdirectory=src"
```
### Importing
```python
from parallel_mc_battery import ParallelMCBattery
```
### TL; DR
`ParallelMCBattery` is a helper class to orchestrate in-parallel Monte Carlo simulations. Its primary role is to abstract away the creation, execution, and management of an efficient Apache Beam pipeline.

### The demo Jupyter notebooks (highly recommended):
1. [Biased and unbiased coin sequences of arbitrary length](demo/demo.ipynb)
### Description
 

`ParallelMCBattery` equips its instances with the attributes corresponding to the values for the *battery configuration dictionary* keys: 
* `rng` --> an __OPTIONAL__ *string* from the choices `"PCG64"`, `"Philox"`, `"SFC64"`, `"MT19937"`, which defaults to `"PCG64"` if omitted
* `pipeline_options` --> an instance of the class [PipelineOptions](https://beam.apache.org/releases/pydoc/2.33.0/apache_beam.options.pipeline_options.html#apache_beam.options.pipeline_options.PipelineOptions) of Apache Beam

Note that the **pipeline options** is the only way the parallel execution can be customized, as the pipeline creation, execution, and management is abstracted away by the `ParallelMCBattery` class.

The Monte Carlo simulations are defined by specifying the values for the *simulation configuration dictionary* keys:
* `models` --> a list of functions that embodies the logic of the model (see Jupyter notebooks below)
* `parameters` --> an __OPTIONAL__ list (possibly of lists) of values of type *float* representing the parameters of each of the model defined in `models`
* `starting point` --> an __OPTIONAL__  *float* representing the start of the Monte Carlo simulation
* `number_simulations` --> an *integer* representing the number of Monte Carlo simulations
* `number_points` --> an *integer* representing the number of points in a Monte Carlo simulation

Moreover, when defining the Monte Carlo simulations, one further specifies the output_paths:
* `output_paths` --> an __OPTIONAL__ list of *string* values representing the *local* paths of the output csv files, and which defaults to the directory where the script with the Monte Carlo simulation is executed (the BigQuery/Google Cloud Storage adaptor currently UNDER DEVELOPMENT)

### General workflow
One may configure the Monte Carlo parallel battery by choosing the desired random number generator, in this case [Philox](https://numpy.org/doc/stable/reference/random/bit_generators/philox.html#philox-counter-based-rng), and the pipeline options instance of the class [PipelineOptions](https://beam.apache.org/releases/pydoc/2.33.0/apache_beam.options.pipeline_options.html#apache_beam.options.pipeline_options.PipelineOptions) of Apache Beam, for this example choosing for simplicity the default settings, which means the pipeline runs on the local `Direct Runner`.
```python
options = PipelineOptions()

battery_configs = {"rng" : "Philox", "pipeline_options": options}
battery_parallel_MC = ParallelMCBattery(battery_configs=battery_configs)
```

The following step is to define the models. 

Example: two distinct **parallelizable** sequences of heads 'H' and tails 'T' generated by simulating tossing coins of a given `bias`, which is a list of `parameters`, corresponding in this case to a unique parameter.
```python
def CoinSequence(number_flips, rng, bias):
    return ["H" if rng.random() <= bias[0] else "T" for _ in range(number_flips)]

models = [CoinSequence, CoinSequence]
```

To configure the simulations for these models, one further uses a list of dictionaries, each dictionary corresponding to one of the `models`.
```python
# 100,000 simulations of sequences of 3 heads or tails, for an unbiased coin
unbiased_coin_config = {"parameters": [0.5], "number_simulations" : 100000, "number_points": 3}
# 60,000 simulations of sequences of 5 heads or tails, for a biased coin 
biased_coin_config = {"parameters": [0.7], "number_simulations" : 60000, "number_points": 5}

simulation_configs = [unbiased_coin_config, biased_coin_config]
```

And, finally, one may perform the Monte Carlo simulations configured above, having in mind that without specifying `output_paths`, the output csv files will be written by default in the directory where it is executed the Python script which calls the object method `simulate`:
```python
battery_parallel_MC.simulate(models, simulation_configs)
```

