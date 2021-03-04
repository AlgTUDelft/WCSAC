# wcsac 

**Warning:** If you want to use the algorithm WCSAC from the paper, make sure to install Safety Gym according to the instructions on the [Safety Gym repo](https://github.com/openai/safety-gym).
The various utilities here are copied over from [Spinning Up in Deep RL](https://github.com/openai/spinningup/tree/master/spinup/utils). 
Two custom environments (`StaticEnv-v0` and `DynamicEnv-v0`) are created within wcsac.py and saclag.py.

## Installation

Two ways to install this package:

### Direct installation

To install this package:

```
git clone https://github.com/yangqisong/wcsac.git

cd wcsac

pip install -e .
```

### Through [Safety Starter Agents]
Portions of the code in saclag.py and wcsac.py are adapted from [Safety Starter Agents](https://github.com/openai/safety-starter-agents).
So you can install [Safety Starter Agents], and add wcsac.py and saclag.py of our package to /path/to/safety-starter-agents/safe_rl/sac.
Then you can follow the instructions on the [Safety Starter Agents] to use our algorithms as using their given baselines.

## Cite the Paper

If you use WCSAC code in your paper, please cite:

```
@inproceedings{WCSAC,
  title     = {{WCSAC: Worst-Case Soft Actor Critic for Safety-Constrained Reinforcement Learning}},
  author    = {
      Yang, Qisong and 
      Simão, Thiago D. and
      Tindemans, Simon H. and
      Spaan, Matthijs T. J.
  },
  booktitle = {Thirty-Fifth AAAI Conference on Artificial Intelligence},
  year      = {2021},
  note      = {To appear}
}
```


