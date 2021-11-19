from gym.envs.registration import register
register(
    id='SPYENV-v0',
    entry_point='wcsac.sac.SpyUnimodal:SpyUnimodal',
)

register(
    id='SPYENV-v1',
    entry_point='wcsacsac.sac.SpyBimodal:SpyBimodal',
)
