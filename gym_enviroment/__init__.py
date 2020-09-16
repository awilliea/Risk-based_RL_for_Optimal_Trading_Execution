from gym.envs.registration import register

register(
    id='enviroment-v0',
    entry_point='gym_enviroment.envs:ExecutionEnv'
)
