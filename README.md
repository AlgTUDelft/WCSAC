# wcsac 

Portions of the code in saclag.py and wcsac.py are adapted from [Safety Starter Agents](https://github.com/openai/safety-starter-agents).

**Warning:** If you want to use the WCSAC algorithm in [Safety Gym](https://github.com/openai/safety-gym), make sure to install Safety Gym according to the instructions on the [Safety Gym repo](https://github.com/openai/safety-gym).

The various utilities here are copied over from [Spinning Up in Deep RL](https://github.com/openai/spinningup/tree/master/spinup/utils). 
Two custom environments (`StaticEnv-v0` and `DynamicEnv-v0`) are created within wcsac.py and saclag.py.

## Installation & Operation

Two ways to install this package:

### Direct installation

To install this package:

```
git clone https://github.com/yangqisong/wcsac.git

cd wcsac

pip install -e .
```

### Through [Safety Starter Agents]
So you can install [Safety Starter Agents](https://github.com/openai/safety-starter-agents), and add wcsac.py and saclag.py of our package to `/path/to/safety-starter-agents/safe_rl/sac`.
Then you can follow the instructions on the [Safety Starter Agents](https://github.com/openai/safety-starter-agents) to use our algorithms as using their given baselines.

### Reproduce Experiments
**Example:** If you want to test WCSAC with risk level 0.5 in `StaticEnv-v0`, run:
```
cd /path/to/wcsac

python wcsac.py --env 'StaticEnv-v0' -s SEED --cl 0.5 --cost_lim d --logger_kwargs_str '{"output_dir": "./xxx"}'
```
where `SEED` is the random seed, `d` is the real-world safety threshold, and `'{"output_dir": "./xxx"}'` indicates where to store the data. 



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


