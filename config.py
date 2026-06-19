"""Phase 1 → 2 过渡配置"""

# Environment
GRID_SIZE = 12
WALL_DENSITY = 0.1
INTERACTIVE_DENSITY = 0.30
MAX_STEPS_PER_EPISODE = 200
OBS_MODE = 'pixels'        # 'vector' | 'pixels'
VISUAL_CELL_SIZE = 8        # 像素模式：每个格子渲染为 N×N 像素

# Visual Encoder
FEATURE_DIM = 128           # CNN 编码器输出维度
ENCODER_LR = 1e-3

# World Model
WM_HIDDEN_DIM = 48
WM_NUM_LAYERS = 2
WM_LR = 1e-3
ENSEMBLE_SIZE = 3           # 系综世界模型数量

# Policy (Actor-Critic)
POLICY_HIDDEN_DIM = 128
POLICY_LR = 1e-3
GAMMA = 0.99
ENTROPY_COEFF = 0.05

# Curiosity Reward
REWARD_SCALE = 30.0         # 系综分歧缩放
REWARD_MAX = 2.0

# Memory
MEMORY_CAPACITY = 50000
BATCH_SIZE = 64

# Training
NUM_EPISODES = 500
TRAIN_POLICY_EVERY = 50
REPLAY_INTERVAL = 50
REPLAY_STEPS = 5

# Visualization
RENDER = True
