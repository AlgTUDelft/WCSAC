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

A custom environment is created within wcsac.py and saclag.py, which is used in the Empirical Analysis of the paper.
As to hyperparameters and experimental setup, you can refer to the paper and its Appendix.
