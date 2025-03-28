import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
from stable_baselines3 import PPO
from Environment import AberrationCorrectionEnv

# 跟训练时相同的环境
env = AberrationCorrectionEnv(max_steps=10)

# 加载已经训练好的模型
model_path = r"C:\Users\fdl\Desktop\best_model_folder\best_model.zip"
model = PPO.load(model_path, env=env)
print("Model loaded successfully.")

# 在环境中执行推理并渲染
obs, info = env.reset()
done = False
while not done:
    # 渲染当前环境
    env.render()

    # 使用训练好的模型进行决策
    action, _states = model.predict(obs, deterministic=True)

    # 让环境执行这个动作
    obs, reward, terminated, truncated, info = env.step(action)

    # 判断是否 episode 结束
    done = terminated or truncated
    # 你也可以打印奖励，或收集奖励做统计

    print("Reward:", reward)
    print("Action",action)
    print(info['new_norm'])
env.close()
