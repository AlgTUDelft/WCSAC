#Portions of the code are adapted from Safety Starter Agents and Spinning Up, released by OpenAI under the MIT license.
#!/usr/bin/env python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from functools import partial
import numpy as np
import gym
import time
from dark_sac.utils.logx import EpochLogger
from dark_sac.utils.mpi_tf import sync_all_params, MpiAdamOptimizer
from dark_sac.utils.mpi_tools import mpi_fork, mpi_sum, proc_id, mpi_statistics_scalar, num_procs
from safety_gym.envs.engine import Engine
from scipy.stats import norm
import math as m

#only use memory*0.9
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

m_pi = tf.constant(m.pi)

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    x = tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
    return x

def mlp_dis(x, hidden_sizes=(64,), activation=None, layer_norm = True):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h)
        if activation is not None:
            x = activation(x)
    return x

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def get_target_update(main_name, target_name, polyak):
    ''' Get a tensorflow op to update target variables based on main variables '''
    main_vars = {x.name: x for x in get_vars(main_name)}
    targ_vars = {x.name: x for x in get_vars(target_name)}
    assign_ops = []
    for v_targ in targ_vars:
        assert v_targ.startswith(target_name), f'bad var name {v_targ} for {target_name}'
        v_main = v_targ.replace(target_name, main_name, 1)
        assert v_main in main_vars, f'missing var name {v_main}'
        assign_op = tf.assign(targ_vars[v_targ], polyak*targ_vars[v_targ] + (1-polyak)*main_vars[v_main])
        assign_ops.append(assign_op)
    return tf.group(assign_ops)


def get_tau(batch_size, num_quantiles, cvar = 1, tau_type = 'iqn'):
    """
    'fix': quantile fractions are fixed
    'iqn': quantile fraction are randomly generated
    """
    if tau_type == 'fix':
        presum_tau = np.zeros((batch_size, num_quantiles)) + 1. / num_quantiles
    elif tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
        presum_tau = np.random.rand(batch_size, num_quantiles) + 0.1
        presum_tau /= presum_tau.sum(axis=-1, keepdims=True)

    tau = np.cumsum(presum_tau, axis=1)# (N, T), note that they are tau1...tauN in the  paper
    
    tau_hat = np.zeros_like(tau)
    tau_hat[:, 0:1] = tau[:, 0:1] / 2
    tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
    
    #get values for cvar
    tau = 1 - cvar + cvar * tau
    tau_hat = 1 - cvar + cvar * tau_hat
    
    return tau, tau_hat, presum_tau


def _quantile_regression_loss(_input, target, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    _input = tf.expand_dims(_input, axis = -1)
    
    target = tf.expand_dims(target, axis = -2)
    
    tau = tf.expand_dims(tau, axis = -1)
    
    weight = tf.expand_dims(weight, axis = -2)
    
    new_shape_0 = tf.shape(_input)[0]
    new_shape_1 = tf.shape(_input)[1]
    new_shape = [new_shape_0, new_shape_1, new_shape_1]
    
    _input = tf.broadcast_to(_input, new_shape)
    target = tf.broadcast_to(target, new_shape)
    L = tf.losses.huber_loss(_input, target, reduction="none")
    sign = tf.sign(_input - target) / 2. + 0.5
    rho = tf.abs(tau - sign) * L * weight

    return tf.reduce_mean( tf.reduce_sum(rho, axis = -1) )

"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=None)
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


"""
Actors and Critics
"""
def mlp_actor(x, a, name='pi', hidden_sizes=(256,256), activation=tf.nn.relu,
              output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope(name):
        mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale

    return mu, pi, logp_pi

def mlp_critic(x, a, pi, name, hidden_sizes=(256,256), activation=tf.nn.relu,
               output_activation=None, policy=mlp_gaussian_policy, action_space=None):

    fn_mlp = lambda x : tf.squeeze(mlp(x=x,
                                       hidden_sizes=list(hidden_sizes)+[1],
                                       activation=activation,
                                       output_activation=None),
                                   axis=1)
    with tf.variable_scope(name):
        critic = fn_mlp(tf.concat([x,a], axis=-1))

    with tf.variable_scope(name, reuse=True):
        critic_pi = fn_mlp(tf.concat([x,pi], axis=-1))

    return critic, critic_pi


def dist_critic(x, a, pi, tau_hat, tau, name, hidden_sizes=(512,512),activation=tf.nn.relu):
    
    const_vec = tf.range(1, 1 + hidden_sizes[0])
    const_vec = tf.cast(const_vec, tf.float32)

    base_fc = lambda x : mlp_dis(x=x, hidden_sizes = list(hidden_sizes),
                                       activation = activation)
    
    tau_fc = lambda x : mlp_dis(x=x, hidden_sizes = list(hidden_sizes),
                                       activation = tf.nn.sigmoid)
    
    merge_fc = lambda x : mlp_dis(x=x, hidden_sizes = [hidden_sizes[-1], 1],
                                       activation = activation)
    
    last_fc = lambda x : mlp_dis(x=x, hidden_sizes = [1, 1],
                                       activation = None,
                                       layer_norm = None)
    '''
    z_pred ... (x_ph, a_ph, tau_hat)
    zt_pred ... (x_ph, a_ph, tau)
    z_pi ... (x_ph, pi, tau_hat)
    zt_pi ... (x_ph, pi, tau)
    '''
    with tf.variable_scope(name):
        _sa    = base_fc(tf.concat([x, a], axis=1))
        _sa    = tf.expand_dims(_sa, axis = -2)
        
        _tau_hat   = tf.expand_dims(tau_hat, axis = -1)
        _tau_hat   = tf.cos(_tau_hat * const_vec * m_pi)
        _tau_hat   = tau_fc(_tau_hat)
        
        _satauhat = tf.multiply(_sa, _tau_hat)
        _satauhat = merge_fc(_satauhat)
        
        z_pred = last_fc(_satauhat)
        z_pred = tf.squeeze(z_pred, axis = -1)
       
    with tf.variable_scope(name, reuse=True):
        _sa    = base_fc(tf.concat([x, a], axis=1))
        _sa    = tf.expand_dims(_sa, axis = -2)
        
        _tau   = tf.expand_dims(tau, axis = -1)
        _tau   = tf.cos(_tau * const_vec * m_pi)
        _tau  = tau_fc(_tau)
        
        _satau = tf.multiply(_sa, _tau)
        _satau = merge_fc(_satau)
        
        zt_pred = last_fc(_satau)
        zt_pred = tf.squeeze(zt_pred, axis = -1)
        
    with tf.variable_scope(name, reuse=True):
        _spi    = base_fc(tf.concat([x, pi], axis=1))
        _spi    = tf.expand_dims(_spi, axis = -2)
        
        _tau_hat   = tf.expand_dims(tau_hat, axis = -1)
        _tau_hat   = tf.cos(_tau_hat * const_vec * m_pi)
        _tau_hat   = tau_fc(_tau_hat)
        
        _spitauhat = tf.multiply(_spi, _tau_hat)
        _spitauhat = merge_fc(_spitauhat)
        
        z_pi = last_fc(_spitauhat)
        z_pi = tf.squeeze(z_pi, axis = -1)
       
    with tf.variable_scope(name, reuse=True):
        _spi    = base_fc(tf.concat([x, pi], axis=1))
        _spi    = tf.expand_dims(_spi, axis = -2)
        
        _tau   = tf.expand_dims(tau, axis = -1)
        _tau   = tf.cos(_tau * const_vec * m_pi)
        _tau  = tau_fc(_tau)
        
        _spitau = tf.multiply(_spi, _tau)
        _spitau = merge_fc(_spitau)
        
        zt_pi = last_fc(_spitau)
        zt_pi = tf.squeeze(zt_pi, axis = -1)

    return z_pred, zt_pred, z_pi, zt_pi

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.costs_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, cost):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.costs_buf[self.ptr] = cost
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    costs=self.costs_buf[idxs],
                    done=self.done_buf[idxs])


"""
Soft Actor-Critic
"""
def sac(env_fn, actor_fn=mlp_actor, critic_fn=mlp_critic, zdist_fn = dist_critic, quantile_regression_loss = _quantile_regression_loss, ac_kwargs=dict(), seed=0, risk_level = 1.0, num_quantiles=20, steps_per_epoch=1000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-4, batch_size=1024, local_start_steps=int(1e3),
        max_ep_len=1000, logger_kwargs=dict(), save_freq=10, local_update_after=int(1e3),
        update_freq=1, render=False, 
        fixed_entropy_bonus=None, entropy_constraint=-1.0,
        fixed_cost_penalty=None, cost_constraint=None, cost_lim=None,
        reward_scale=1, lr_scale = 1, damp_scale = 0,
        ):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_fn: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the actor
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ===========  ================  ======================================

        critic_fn: A function which takes in placeholder symbols
            for state, ``x_ph``, action, ``a_ph``, and policy ``pi``,
            and returns the critic outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``critic``    (batch,)         | Gives one estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``critic_pi`` (batch,)         | Gives another estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_fn / critic_fn
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        batch_size (int): Minibatch size for SGD.

        local_start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        fixed_entropy_bonus (float or None): Fixed bonus to reward for entropy.
            Units are (points of discounted sum of future reward) / (nats of policy entropy).
            If None, use ``entropy_constraint`` to set bonus value instead.

        entropy_constraint (float): If ``fixed_entropy_bonus`` is None,
            Adjust entropy bonus to maintain at least this much entropy.
            Actual constraint value is multiplied by the dimensions of the action space.
            Units are (nats of policy entropy) / (action dimenson).

        fixed_cost_penalty (float or None): Fixed penalty to reward for cost.
            Units are (points of discounted sum of future reward) / (points of discounted sum of future costs).
            If None, use ``cost_constraint`` to set penalty value instead.

        cost_constraint (float or None): If ``fixed_cost_penalty`` is None,
            Adjust cost penalty to maintain at most this much cost.
            Units are (points of discounted sum of future costs).
            Note: to get an approximate cost_constraint from a cost_lim (undiscounted sum of costs),
            multiply cost_lim by (1 - gamma ** episode_len) / (1 - gamma).
            If None, use cost_lim to calculate constraint.

        cost_lim (float or None): If ``cost_constraint`` is None,
            calculate an approximate constraint cost from this cost limit.
            Units are (expectation of undiscounted sum of costs in a single episode).
            If None, cost_lim is not used, and if no cost constraints are used, do naive optimization.
    """
    use_costs = fixed_cost_penalty or cost_constraint or cost_lim

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Env instantiation
    env, test_env = env_fn(), env_fn()
        
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Setting seeds
    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    test_env.seed(seed)

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph, c_ph = placeholders(obs_dim, act_dim, obs_dim, None, None, None)
    tau_hat_ph, next_tau_hat_ph, next_presum_tau_ph, tau_ph, presum_tau_ph = placeholders(num_quantiles, num_quantiles, num_quantiles, num_quantiles, num_quantiles)
    
    
    local_batch_size = batch_size // num_procs()
    #_, tau, presum_tau = get_tau(local_batch_size, num_quantiles, tau_type = 'fix')
    #tau = tau.astype(np.float32)
    
    # Main outputs from computation graph
    '''
    z_pred ... (x_ph, a_ph, tau_hat)
    zt_pred ... (x_ph, a_ph, tau)
    z_pi ... (x_ph, pi, tau_hat)
    zt_pi ... (x_ph, pi, tau)
    '''
    with tf.variable_scope('main'):
        mu, pi, logp_pi = actor_fn(x_ph, a_ph, **ac_kwargs)
        qr1, qr1_pi = critic_fn(x_ph, a_ph, pi, name='qr1', **ac_kwargs)
        qr2, qr2_pi = critic_fn(x_ph, a_ph, pi, name='qr2', **ac_kwargs)
        qz_pred, qzt_pred, _, qzt_pi = zdist_fn(x_ph, a_ph, pi, tau_hat_ph, tau_ph, name='qz')
        
    with tf.variable_scope('main', reuse=True):
        # Additional policy output from a different observation placeholder
        # This lets us do separate optimization updates (actor, critics, etc)
        # in a single tensorflow op.
        _, pi2, logp_pi2 = actor_fn(x2_ph, a_ph, **ac_kwargs)
    
    # Target value network
    with tf.variable_scope('target'):
        _, qr1_pi_targ = critic_fn(x2_ph, a_ph, pi2,  name='qr1', **ac_kwargs)
        _, qr2_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr2', **ac_kwargs)
        _, _, qz_pi_targ, _  = zdist_fn(x2_ph, a_ph, pi2, next_tau_hat_ph, tau_ph, name='qz')

    # Entropy bonus
    if fixed_entropy_bonus is None:
        with tf.variable_scope('entreg'):
            soft_alpha = tf.get_variable('soft_alpha',
                                         initializer=0.0,
                                         trainable=True,
                                         dtype=tf.float32)
        alpha = tf.nn.softplus(soft_alpha)
    else:
        alpha = tf.constant(fixed_entropy_bonus)
    log_alpha = tf.log(tf.clip_by_value(alpha,1e-8,1e8))

    # Cost penalty
    if use_costs:
        if fixed_cost_penalty is None:
            with tf.variable_scope('costpen'):
                soft_beta = tf.get_variable('soft_beta',
                                             initializer=0.0,
                                             trainable=True,
                                             dtype=tf.float32)
            beta = tf.nn.softplus(soft_beta)
            log_beta = tf.log(tf.clip_by_value(beta,1e-8,1e8))
        else:
            beta = tf.constant(fixed_cost_penalty)
            log_beta = tf.log(tf.clip_by_value(beta,1e-8,1e8))
    else:
        beta = 0.0  # costs do not contribute to policy optimization
        print('Not using costs')

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    if proc_id()==0:
        var_counts = tuple(count_vars(scope) for scope in 
                           ['main/pi', 'main/qr1', 'main/qr2', 'main'])
        print(('\nNumber of parameters: \t pi: %d, \t qr1: %d, \t qr2: %d, \t total: %d\n')%var_counts)
    
    # Min Double-Q:
    min_q_pi = tf.minimum(qr1_pi, qr2_pi)
    min_q_pi_targ = tf.minimum(qr1_pi_targ, qr2_pi_targ)
     
    q_backup = tf.stop_gradient( r_ph + gamma * (1 - d_ph) * (min_q_pi_targ - alpha * logp_pi2))
    c_dis = tf.expand_dims(c_ph, axis=1)
    d_dis = tf.expand_dims(d_ph, axis=1)
    qz_target = tf.stop_gradient( c_dis + gamma * (1 - d_dis) * qz_pi_targ )
    
    '''avoid None value'''
    qzt_pred = tf.clip_by_value(qzt_pred, 1e-8, 1e8)
    qzt_pi = tf.clip_by_value(qzt_pi, 1e-8, 1e8)
    qz_pi_targ = tf.clip_by_value(qz_pi_targ, 1e-8, 1e8)
    qz_target = tf.clip_by_value(qz_target, 1e-8, 1e8)

    # Soft actor-critic losses
    qc = tf.reduce_sum(presum_tau_ph * qzt_pred, axis=1, keepdims=True)
    qc_pi = tf.reduce_sum(presum_tau_ph * qzt_pi, axis=1, keepdims=True)
    
    cost_constraint = cost_lim * (1 - gamma ** max_ep_len) / (1 - gamma) / max_ep_len
    damp = damp_scale * tf.reduce_mean(cost_constraint - qc)

    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi + (beta - damp) * qc_pi)

    qr1_loss = 0.5 * tf.reduce_mean((q_backup - qr1)**2)
    qr2_loss = 0.5 * tf.reduce_mean((q_backup - qr2)**2)
    qzf_loss = quantile_regression_loss(qz_pred, qz_target, tau_hat_ph, next_presum_tau_ph)
    q_loss = qr1_loss + qr2_loss + qzf_loss

    # Loss for alpha
    entropy_constraint *= act_dim
    pi_entropy = -tf.reduce_mean(logp_pi)
    # alpha_loss = - soft_alpha * (entropy_constraint - pi_entropy)
    alpha_loss = - alpha * (entropy_constraint - pi_entropy)
    print('using entropy constraint', entropy_constraint)

    # Loss for beta
    if use_costs:
        if cost_constraint is None:
            # Convert assuming equal cost accumulated each step
            # Note this isn't the case, since the early in episode doesn't usually have cost,
            # but since our algorithm optimizes the discounted infinite horizon from each entry
            # in the replay buffer, we should be approximately correct here.
            # It's worth checking empirical total undiscounted costs to see if they match.
            cost_constraint = cost_lim * (1 - gamma ** max_ep_len) / (1 - gamma) / max_ep_len
        print('using cost constraint', cost_constraint)
        beta_loss = beta * (cost_constraint - qc)

    # Policy train op
    # (has to be separate from value train op, because qr1_pi appears in pi_loss)
    train_pi_op = MpiAdamOptimizer(learning_rate=lr).minimize(pi_loss, var_list=get_vars('main/pi'), name='train_pi')

    # Value train op
    with tf.control_dependencies([train_pi_op]):
        train_q_op = MpiAdamOptimizer(learning_rate=lr).minimize(q_loss, var_list=get_vars('main/q'), name='train_q')

    if fixed_entropy_bonus is None:
        entreg_optimizer = MpiAdamOptimizer(learning_rate=lr)
        with tf.control_dependencies([train_q_op]):
            train_entreg_op = entreg_optimizer.minimize(alpha_loss, var_list=get_vars('entreg'))

    if use_costs and fixed_cost_penalty is None:
        costpen_optimizer = MpiAdamOptimizer(learning_rate=lr*lr_scale)
        if fixed_entropy_bonus is None:
            with tf.control_dependencies([train_entreg_op]):
                train_costpen_op = costpen_optimizer.minimize(beta_loss, var_list=get_vars('costpen'))
        else:
            with tf.control_dependencies([train_q_op]):
                train_costpen_op = costpen_optimizer.minimize(beta_loss, var_list=get_vars('costpen'))
            

    # Polyak averaging for target variables
    target_update = get_target_update('main', 'target', polyak)

    # Single monolithic update with explicit control dependencies
    with tf.control_dependencies([train_pi_op]):
        with tf.control_dependencies([train_q_op]):
            grouped_update = tf.group([target_update])

    if fixed_entropy_bonus is None:
        grouped_update = tf.group([grouped_update, train_entreg_op])
    if use_costs and fixed_cost_penalty is None:
        grouped_update = tf.group([grouped_update, train_costpen_op])

    # Initializing targets to match main variables
    # As a shortcut, use our exponential moving average update w/ coefficient zero
    target_init = get_target_update('main', 'target', 0.0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                                outputs={'mu': mu, 'pi': pi, 'qr1': qr1, 'qr2': qr2, 'qc': qc})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]
        
    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_cost, ep_len, ep_goals, = test_env.reset(), 0, False, 0, 0, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, info = test_env.step(get_action(o, True))
                if render and proc_id() == 0 and j == 0:
                    test_env.render()
                ep_ret += r
                ep_cost += info.get('cost', 0)
                ep_len += 1
                ep_goals += 1 if info.get('goal_met', False) else 0
            logger.store(TestEpRet=ep_ret, TestEpCost=ep_cost, TestEpLen=ep_len, TestEpGoals=ep_goals)

    start_time = time.time()
    o, r, d, ep_ret, ep_cost, ep_len, ep_goals = env.reset(), 0, False, 0, 0, 0, 0
    total_steps = steps_per_epoch * epochs

    # variables to measure in an update
    vars_to_get = dict(LossPi=pi_loss, LossQR1=qr1_loss, LossQR2=qr2_loss, LossZF=qzf_loss,
                       QR1Vals=qr1, QR2Vals=qr2, QCVals = qc, LogPi=logp_pi, PiEntropy=pi_entropy, Alpha=alpha, LogAlpha=log_alpha, LossAlpha=alpha_loss)
    if use_costs:
        vars_to_get.update(dict(Beta=beta, LogBeta=log_beta, LossBeta=beta_loss))

    print('starting training', proc_id())

    # Main loop: collect experience in env and update/log each epoch
    number_model = 0
    cum_cost = 0
    local_steps = 0
    local_steps_per_epoch = steps_per_epoch // num_procs()
    #local_batch_size = batch_size // num_procs()
    epoch_start_time = time.time()
    for t in range(total_steps // num_procs()):
        """
        Until local_start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy.
        """
        if t > local_start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        r *= reward_scale  # yee-haw
        c = info.get('cost', 0)
        ep_ret += r
        ep_cost += c
        ep_len += 1
        ep_goals += 1 if info.get('goal_met', False) else 0
        local_steps += 1
        
        # Track cumulative cost over training
        cum_cost += c

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, c)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len, EpGoals=ep_goals)
            o, r, d, ep_ret, ep_cost, ep_len, ep_goals = env.reset(), 0, False, 0, 0, 0, 0

        if t > 0 and t % update_freq == 0:
            #if index_risk < 0:
            #    index_risk = 0
                
            for j in range(update_freq):
                #batch = replay_buffer.sample_batch(local_batch_size)
                batch = replay_buffer.sample_batch(local_batch_size)
                _, tau_hat, _ = get_tau(local_batch_size, num_quantiles)
                _, next_tau_hat, next_presum_tau = get_tau(local_batch_size, num_quantiles)
                _, tau, presum_tau = get_tau(local_batch_size, num_quantiles, cvar = risk_level)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             c_ph: batch['costs'],
                             d_ph: batch['done'],
                             tau_hat_ph: tau_hat,
                             next_tau_hat_ph: next_tau_hat,
                             next_presum_tau_ph: next_presum_tau,
                             tau_ph: tau,
                             presum_tau_ph: presum_tau,
                            }
                if t < local_update_after:
                    logger.store(**sess.run(vars_to_get, feed_dict))
                else:
                    values, _ = sess.run([vars_to_get, grouped_update], feed_dict)
                    logger.store(**values)

        # End of epoch wrap-up
        if t > 0 and t % local_steps_per_epoch == 0:
            epoch = t // local_steps_per_epoch
            
            #=====================================================================#
            #  Cumulative cost calculations                                       #
            #=====================================================================#
            cumulative_cost = mpi_sum(cum_cost)
            cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)
            

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, number_model)
                number_model += 1

            # Test the performance of the deterministic version of the agent.
            test_start_time = time.time()
            test_agent()
            logger.store(TestTime=time.time() - test_start_time)

            logger.store(EpochTime=time.time() - epoch_start_time)
            epoch_start_time = time.time()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('TestEpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('EpGoals', average_only=True)
            logger.log_tabular('TestEpGoals', average_only=True)
            logger.log_tabular('CumulativeCost', cumulative_cost)
            logger.log_tabular('CostRate', cost_rate)
            logger.log_tabular('TotalEnvInteracts', mpi_sum(local_steps))
            logger.log_tabular('QR1Vals', with_min_and_max=True)
            logger.log_tabular('QR2Vals', with_min_and_max=True)
            logger.log_tabular('QCVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQR1', average_only=True)
            logger.log_tabular('LossQR2', average_only=True)
            logger.log_tabular('LossZF', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('LogAlpha', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            if use_costs:
                logger.log_tabular('LossBeta', average_only=True)
                logger.log_tabular('LogBeta', average_only=True)
                logger.log_tabular('Beta', average_only=True)
            logger.log_tabular('PiEntropy', average_only=True)
            logger.log_tabular('TestTime', average_only=True)
            logger.log_tabular('EpochTime', average_only=True)
            logger.log_tabular('TotalTime', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--rl', type=float, default=0.5)
    parser.add_argument('--nq', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--steps_per_epoch', type=int, default=30000)
    parser.add_argument('--update_freq', type=int, default=100)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--local_start_steps', default=500, type=int)
    parser.add_argument('--local_update_after', default=500, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--fixed_entropy_bonus', default=None, type=float)
    parser.add_argument('--entropy_constraint', type=float, default= -1)
    parser.add_argument('--fixed_cost_penalty', default=None, type=float)
    parser.add_argument('--cost_constraint', type=float, default=None)
    parser.add_argument('--cost_lim', type=float, default=25)
    parser.add_argument('--lr_s', type=int, default=20)
    parser.add_argument('--damp_s', type=int, default=10)
    parser.add_argument('--logger_kwargs_str', type=json.loads, default='{"output_dir": "./data"}')
    args = parser.parse_args()

    try:
        import safety_gym
    except:
        print('Make sure to install Safety Gym to use constrained RL environments.')

    from dark_sac.utils.run_utils import setup_logger_kwargs
    
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs= args.logger_kwargs_str

    sac(lambda : gym.make(args.env), actor_fn=mlp_actor, critic_fn=mlp_critic, zdist_fn = dist_critic, quantile_regression_loss = _quantile_regression_loss, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, risk_level=args.rl, num_quantiles=args.nq, epochs=args.epochs, batch_size=args.batch_size,
        logger_kwargs=logger_kwargs, steps_per_epoch=args.steps_per_epoch,
        update_freq=args.update_freq, lr=args.lr, render=args.render,
        local_start_steps=args.local_start_steps, local_update_after=args.local_update_after,
        fixed_entropy_bonus=args.fixed_entropy_bonus, entropy_constraint=args.entropy_constraint,
        fixed_cost_penalty=args.fixed_cost_penalty, cost_constraint=args.cost_constraint, cost_lim = args.cost_lim, lr_scale = args.lr_s, damp_scale = args.damp_s,
        )
