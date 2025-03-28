import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from PPO_structure import CustomCombinedExtractor
from Environment import AberrationCorrectionEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = AberrationCorrectionEnv(max_steps=15)
check_env(env)


policy_kwargs = {
    "features_extractor_class": CustomCombinedExtractor,
    "features_extractor_kwargs": {}
}

model = PPO(
    "MultiInputPolicy", env,
    batch_size=15,
    learning_rate=0.007,
    n_steps=200,
    gamma=0.995,
    gae_lambda=0.95,
    ent_coef=0.15,
    verbose=1,
    vf_coef=2,
    clip_range=0.2,
    policy_kwargs=policy_kwargs
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=r"C:\Users\fdl\Desktop\best_model_folder",
    log_path=r"C:\Users\fdl\Desktop\results",
    eval_freq=150,
    deterministic=True,
    render=False
)

model.learn(total_timesteps=250000, callback=eval_callback)

'''
model = PPO.load("ppo_aberration_correction", env=env)  
print("Model loaded successfully.")
'''

obs, info = env.reset()
done = True
while not done:
    env.render()  # 渲染当前状态
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print("Reward:", reward)
env.close()

