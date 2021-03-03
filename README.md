# wcsac 

If you want to use the algorithm WCSAC from the paper, make sure to install Safety Gym according to the instructions on the [Safety Gym repo](https://github.com/openai/safety-gym).
The various utilities here are copied over from [Spinning Up in Deep RL](https://github.com/openai/spinningup/tree/master/spinup/utils). 

Two ways to install this package:

(1) Direct installation
To install this package:
cd /path/to/wcsac
pip install -e .

(2) Through [Safety Starter Agents]
Portions of the code in saclag.py and wcsac.py are adapted from [Safety Starter Agents](https://github.com/openai/safety-starter-agents).
So you can install [Safety Starter Agents], and add wcsac.py and saclag.py of our package to /path/to/safety-starter-agents/safe_rl/sac.
Then you can follow the instructions on the [Safety Starter Agents] to use our algorithms as using their given baselines.

To reproduce an experiment from the paper, run:

#WCSAC
python /path/to/wcsac/sac/wcsac.py --cl 0.1/0.5/0.9 --seed 0/10/20 --epochs 100 --exp_name xxx --steps_per_epoch 30000 --cost_lim 15 --logger_kwargs_str '{"output_dir": "./xxx"}'

#SAC-Lagrangian
python /path/to/wcsac/sac/saclag.py --seed 0/10/20 --epochs 100 --exp_name xxx --steps_per_epoch 30000 --cost_lim 15 --logger_kwargs_str '{"output_dir": "./xxx"}'

#SAC
python /path/to/wcsac/sac/saclag.py --seed 0/10/20 --epochs 100 --exp_name xxx --steps_per_epoch 30000 --logger_kwargs_str '{"output_dir": "./xxx"}'

A custom environment is created within wcsac.py and saclag.py, which is used in the Empirical Analysis of the paper.
As to hyperparameters and experimental setup, you can refer to the paper and its Appendix.
