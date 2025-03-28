import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pleasenet import new


class CustomCombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64):
        # observation_space 应包含 "old_image", "new_image", "action"
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        self.Net = new(num_classes=4)
        combined_input_dim = 4 + 4 + 4
        self.fc = nn.Sequential(
            nn.Linear(combined_input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, features_dim),
            nn.ReLU()
        )

    def image2tensor(self,img_arr):
        img_tensor = torch.from_numpy(img_arr).float()  # 直接转换为 float32
        mean_tensor = img_tensor.mean(dim=(1, 2), keepdim=True)
        std_tensor = img_tensor.std(dim=(1, 2), keepdim=True)
        img_tensor = (img_tensor - mean_tensor) / (std_tensor + 1e-6)
        return img_tensor

    def forward(self, observations: dict) -> torch.Tensor:
        """
        observations 是一个字典，包含 "old_image", "new_image" 和 "action" 三部分数据。
        其中旧图像和新图像形状均为 (batch, C, H, W)，动作 shape 为 (batch, 4)
        obervations返回的结果都是以tensor的形式存储的
        """
        old_image = observations["old_image"]
        new_image = observations["new_image"]
        action = observations["action"]

        self.Net.load_state_dict(torch.load(r"E:\previous-version\new_simpler.pth", map_location=self.device))
        self.Net.to(self.device)
        self.Net.eval()
        self.Net.float()

        with torch.no_grad():
            old_PSF = self.Net(old_image)
            new_PSF = self.Net(new_image)

        # 拼接旧图像和新图像的 PSF 预测结果以及动作，得到 (batch, 12) 的特征向量
        combined = torch.cat([old_PSF, new_PSF, action], dim=1)

        # 经过全连接网络提取最终特征表示
        features = self.fc(combined)
        return features
