# Parallelize Monte Carlo simulations with Apache Beam
## ParallelMCBattery class
* Helper class to orchestrate in parallel Monte Carlo simulations
* Instance with intuitive attributes such as `models`, `parameters`, `number_simulations` etc. allow to create an efficient Apache Beam pipeline with minimum low-level tuning
### Example
```python
battery_configs = {"rng" : "Philox", "pipeline_options": options}

battery_parallel_MC = ParallelMCBattery(battery_configs=battery_configs)

unbiased_coin_config = {"number_simulations" : 100000, "number_points": 3, "parameters": [0.5]}
biased_coin_config = {"number_simulations" : 60000, "number_points": 5, "parameters": [0.7]}

simulation_configs = [unbiased_coin_config, biased_coin_config]

battery_parallel_MC.simulate(models, simulation_configs)
```
### SEE THE DEMO JUPYTER NOTEBOOKS:
1. [Biased and unbiased coins](demo/demo.ipynb)
