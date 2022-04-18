from gym.envs.registration import register
from .NasEnv import *

register(
    id='NAS-Env-v0',
    entry_point='envs:NasEnv',
)
