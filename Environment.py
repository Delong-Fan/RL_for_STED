# environment.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
from gymnasium.core import Env
import numpy as np
import gymnasium as gym
import torch
from torch.utils.data import Dataset
import numpy.fft as fft


class Single_image_Dataloader(Dataset):
    # 用来构造Dataloader,以便能扔进神经网络里
    def __init__(self, img_tensor):
        self.img_tensor = img_tensor

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.img_tensor


class AberrationCorrectionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, image_path=None, max_steps=15, render_mode="human", **kwargs):
        super().__init__()
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        # 连续动作空间：可以定义不同的像差调整范围
        low = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)

        # 观测空间：返回图像+PSF
        self.observation_space = gym.spaces.Dict({
            "old_image": gym.spaces.Box(low=0.0, high=1.0, shape=(5, 256, 256), dtype=np.float32),
            "new_image": gym.spaces.Box(low=0.0, high=1.0, shape=(5, 256, 256), dtype=np.float32),
            "action": gym.spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)
        })

        self.max_steps = max_steps
        self.current_step = 0
        self.image_path = r"E:\Mini_try"
        self.img_files = []

        self.render_mode = kwargs.get('render_mode', 'human')
        self.image_dir = kwargs.get('imageDir', './images')

        self.current_image = None
        self.current_tensor = None
        self.model_path = r"E:\previous-version\new_simpler.pth"
        self.window = None
        self.clock = None

        # Initialised parameters:
        self.NA = 1.4
        self.emission = 1.0
        self.pixelsize = 15
        self.xysize = 256
        self.z_steps = 5
        self.step_size = 5
        self.current_aberration = np.zeros(4)
        self.STED=False

    def draw_beads(self):
        '''
        Makes a fake bead sample
        '''
        img = np.random.rand(256, 256)
        img[img < 0.999] = 0
        img[img >= 0.999] = 1

        return img

    def calcPSF(self):
        '''
        Calculate the PSF of the current imaging parameters
        output:
        psf: 3D numpy array, PSF
        '''

        dkxy = 2 * np.pi / (self.pixelsize * self.xysize)
        kMax = (2 * np.pi * self.NA) / (self.emission * dkxy)
        klims = np.linspace(-self.xysize / 2, self.xysize / 2, self.xysize)
        kx, ky = np.meshgrid(klims, klims)
        k = np.hypot(kx, ky)
        pupil = np.copy(k)
        pupil[pupil < kMax] = 1
        pupil[pupil >= kMax] = 0

        rho_unnormalization = np.sqrt(kx ** 2 + ky ** 2)
        rho = rho_unnormalization / np.amax(rho_unnormalization)
        phi = np.arctan2(ky, kx)

        astig_1 = self.current_aberration[0]
        astig_2 = self.current_aberration[1]
        coma_1 = self.current_aberration[2]
        coma_2 = self.current_aberration[3]

        astig_1_mask = astig_1 * np.sqrt(6) * rho ** 2 * np.sin(2 * phi)
        astig_2_mask = astig_2 * np.sqrt(6) * rho ** 2 * np.cos(2 * phi)
        coma_1_mask = coma_1 * np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.sin(phi)
        coma_2_mask = coma_2 * np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.cos(phi)
        # spherical_mask = self.spherical * np.sqrt(5) * (6 * rho ** 4 - 6 * rho ** 2 + 1)
        # phase_mask = (astig_1_mask + astig_2_mask + coma_1_mask + coma_2_mask + spherical_mask) * 2 * np.pi
        phase_mask = (astig_1_mask + astig_2_mask + coma_1_mask + coma_2_mask) * 2 * np.pi

        psf = np.zeros((self.z_steps, self.xysize, self.xysize))

        for i in range(self.z_steps):
            z = i * self.step_size - (self.z_steps - 1) * self.step_size / 2
            focus_phase = z * np.sqrt(3) * (2 * (rho ** 2) - 1)
            if self.STED:
                spiral_phase = np.pi*2*(phi/np.amax(phi))
            else:
                spiral_phase=0

            pupil_correction = pupil * np.exp(1j * np.mod(phase_mask + focus_phase + spiral_phase, 2 * np.pi))

            psf[i, :, :] = np.abs(fft.ifft2(fft.ifftshift(pupil_correction))) ** 2

        return psf

    def PSF_conv(self, img_arr, PSF):
        '''
        Performs the N-dimensional convolution of the input image with the PSF
        input:
        img_arr: N-dimensional numpy array, input image
        PSF: N-dimensional numpy array, PSF
        output:
        image: N-dimensional numpy array, convolved image
        '''

        OTF = fft.fftn(np.fft.fftshift(PSF))
        image = fft.ifftn(fft.fftn(np.fft.fftshift(img_arr)) * OTF)

        image = np.real(image)
        image = image - np.amin(image)
        image = image / np.amax(image)

        return image

    def image2tensor(self, img_arr):
        img_tensor = torch.from_numpy(img_arr).float()
        mean_tensor = img_tensor.mean(dim=(1, 2), keepdim=True)
        std_tensor = img_tensor.std(dim=(1, 2), keepdim=True)
        img_tensor = (img_tensor - mean_tensor) / (std_tensor + 1e-6)

        return img_tensor

    def reset(self, seed=1):
        '''
        Resets the environment to a new random state
        Initialise the imaging parameters, the sample and the aberrations
        '''
        self.current_step = 0

        # New random imaging parameters
        # For the moment there is very little randomisation to make things easy

        self.NA = 1.4
        self.emission = np.random.randint(550, 551)
        self.pixelsize = np.random.randint(50, 51)

        # New random bead sample
        self.sample = self.draw_beads()

        # Current aberrations of the microscope in an array of form:
        # [astig_1, astig_2, coma_1, coma_2]

        self.current_aberration = random_action = np.random.uniform(low=-2.5, high=2.5, size=(4,)).astype(np.float32)

        # Calculate the PSF of the current imaging parameters
        PSF = self.calcPSF()

        # Convolve the sample with the PSF
        self.current_image = self.PSF_conv(self.sample, PSF)

        # Initial random action. Could also be set to zero which works for the focus environment
        random_action = np.random.uniform(low=-0.5, high=0.5, size=(4,)).astype(np.float32)

        # New aberration is the sum of the current aberration and the action
        self.current_aberration = self.current_aberration + random_action

        # Convolve the sample with the new PSF
        new_image = self.PSF_conv(self.sample, PSF)

        # Make the observation
        obs = {
            "old_image": self.current_image.astype(np.float32),
            "new_image": new_image.astype(np.float32),
            "action": random_action.astype(np.float32)
        }
        info = {}

        # Update the current image
        self.current_image = new_image

        return obs, info

    def step(self, action):
        '''
        Takes a step in the environment based on the input action from the agent
        '''

        # New aberration is the sum of the current aberration and the action
        old_norm = np.sum(abs(self.current_aberration))
        self.current_aberration = self.current_aberration + action

        # Convolve the sample with the new PSF
        new_image = self.PSF_conv(self.sample, self.calcPSF())

        # Initial values
        terminated = False
        truncated = False

        # Calculate the reward as the negative of the distance from [0,0,0,0] aberration
        new_norm = np.sum(abs(self.current_aberration))
        current_step_reward = -new_norm
        historical_reward = (old_norm-new_norm)*6
        reward = (2*current_step_reward + historical_reward)*1

        if old_norm - new_norm > 0.5:
            reward += 10
        if old_norm - new_norm > 1:
            reward += 25

        if new_norm - old_norm > 1:
            reward -= 10
        # Update the step count. Could be used to penalise the agent for taking too many steps later

        # Terminate if the aberration is too large or too small
        if new_norm > 12:
            reward -= 30

        if new_norm < 0.6:
            terminated = True
            reward += 40

            # Truncate if the agent takes too many steps
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 15

        obs = {
            "old_image": self.current_image.astype(np.float32),
            "new_image": new_image.astype(np.float32),
            "action": action.astype(np.float32)
        }
        info = {"new_norm": new_norm}
        self.current_image = new_image
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    def _render_frame(self):
        # 如果窗口尚未创建，则初始化 pygame 并创建窗口
        if self.window is None and self.render_mode == "human":
            import pygame
            pygame.init()
            pygame.display.init()
            self.size = 256  # 假设图像尺寸为 256x256
            self.window = pygame.display.set_mode((self.size, self.size))
        if self.clock is None and self.render_mode == "human":
            import pygame
            self.clock = pygame.time.Clock()

        import pygame
        # 取当前图像的第一个通道（假设它是灰度图）
        img = self.current_image[2, :, :]
        img = (img * 255).astype(np.uint8)

        # 将单通道灰度图复制到三个通道中，生成形状为 (256, 256, 3) 的数组
        img_gray = np.stack([img, img, img], axis=-1)

        # 由于 pygame.surfarray.make_surface 需要数组形状为 (width, height, channels)
        # 我们需要将数组轴交换（注意：通常图像数组形状为 (height, width, channels)）
        img_surface = pygame.surfarray.make_surface(img_gray.swapaxes(0, 1))

        self.window.blit(img_surface, (0, 0))
        pygame.display.flip()
        pygame.event.pump()
        pygame.display.update()

        # 控制渲染帧率
        self.clock.tick(self.metadata["render_fps"])

    def render(self, mode="human"):
        if mode == "human":
            self._render_frame()

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()


if '__main__' == __name__:
    import matplotlib.pyplot as plt

    env = AberrationCorrectionEnv()

    obs = env.reset()

    for i in range(15):
        env.render()
        action = 2.5 - 5 * np.random.rand(4)  # Take random actions during testing
        print(action)
        obs, _, _, _, info = env.step(action)
        env._render_frame()
        if np.mod(i,5):
            env.reset()
    env.close()
