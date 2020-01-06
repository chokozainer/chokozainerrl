from gym.envs.registration import register

register(
    id='Test-v0',
    entry_point='chokozainer.envs.test:MyEnv'
)