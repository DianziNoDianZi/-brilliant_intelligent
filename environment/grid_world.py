import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Cell types
EMPTY = 0
WALL = 1
FOLDER_CLOSED = 2
FOLDER_OPEN = 3
BUTTON_UP = 4
BUTTON_DOWN = 5
DOOR_CLOSED = 6
DOOR_OPEN = 7
NUM_CELL_TYPES = 8

# Actions
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
INTERACT = 4
NUM_ACTIONS = 5

INTERACTIVE_TYPES = {FOLDER_CLOSED, BUTTON_UP, DOOR_CLOSED}

# RGB colors for rendering (BGR for OpenCV compatibility stored as RGB)
CELL_PALETTE = {
    EMPTY: (240, 240, 240),
    WALL: (50, 50, 50),
    FOLDER_CLOSED: (180, 140, 40),
    FOLDER_OPEN: (255, 200, 50),
    BUTTON_UP: (80, 130, 200),
    BUTTON_DOWN: (40, 80, 200),
    DOOR_CLOSED: (160, 100, 40),
    DOOR_OPEN: (220, 170, 80),
}


class GridWorld(gym.Env):
    """Grid world with vector or pixel observations."""

    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(self, grid_size=12, wall_density=0.1,
                 interactive_density=0.18, max_steps=200,
                 obs_mode='vector'):
        super().__init__()
        self.grid_size = grid_size
        self.wall_density = wall_density
        self.interactive_density = interactive_density
        self.max_steps = max_steps
        self.obs_mode = obs_mode
        self.step_count = 0

        self.action_space = spaces.Discrete(NUM_ACTIONS)

        if obs_mode == 'vector':
            obs_dim = grid_size * grid_size + 2
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
        elif obs_mode == 'pixels':
            self.cell_size = 8  # default, overridable via options
            px = grid_size * self.cell_size
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(px, px, 3), dtype=np.uint8
            )

        self._generate_grid()

    def _generate_grid(self):
        size = self.grid_size
        self.grid = np.zeros((size, size), dtype=np.int32)

        wall_mask = np.random.random((size, size)) < self.wall_density
        self.grid[wall_mask] = WALL

        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        empty_cells = list(zip(*np.where(self.grid == EMPTY)))
        np.random.shuffle(empty_cells)

        num_interactive = max(1, int(len(empty_cells) * self.interactive_density))
        types = [FOLDER_CLOSED, BUTTON_UP, DOOR_CLOSED]
        for i, (r, c) in enumerate(empty_cells[:num_interactive]):
            self.grid[r, c] = types[i % len(types)]

        remaining_empty = list(zip(*np.where(self.grid == EMPTY)))
        if remaining_empty:
            idx = np.random.randint(len(remaining_empty))
            self.agent_pos = list(remaining_empty[idx])
        else:
            self.agent_pos = [size // 2, size // 2]
            self.grid[self.agent_pos[0], self.agent_pos[1]] = EMPTY

    def _get_obs(self):
        if self.obs_mode == 'vector':
            size = self.grid_size
            grid_flat = self.grid.flatten().astype(np.float32) / (NUM_CELL_TYPES - 1)
            pos = np.array([
                self.agent_pos[0] / (size - 1),
                self.agent_pos[1] / (size - 1)
            ], dtype=np.float32)
            return np.concatenate([grid_flat, pos])
        else:
            return self._render_pixels(self.cell_size)

    def _render_pixels(self, cell_size):
        """Render the grid as an RGB image array."""
        size = self.grid_size
        px = size * cell_size
        img = np.zeros((px, px, 3), dtype=np.uint8)

        for r in range(size):
            for c in range(size):
                color = CELL_PALETTE.get(self.grid[r, c], (200, 200, 200))
                y0, y1 = r * cell_size, (r + 1) * cell_size
                x0, x1 = c * cell_size, (c + 1) * cell_size
                img[y0:y1, x0:x1] = color

        # Draw agent as a filled circle
        ar, ac = self.agent_pos
        cy = int(ar * cell_size + cell_size // 2)
        cx = int(ac * cell_size + cell_size // 2)
        radius = max(cell_size // 2 - 2, 1)
        self._draw_circle(img, cx, cy, radius, (220, 50, 50))

        return img

    def _draw_circle(self, img, cx, cy, r, color):
        """Bresenham-style filled circle on numpy array."""
        for y in range(max(0, cy - r), min(img.shape[0], cy + r + 1)):
            for x in range(max(0, cx - r), min(img.shape[1], cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    img[y, x] = color

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'cell_size' in options:
            self.cell_size = options['cell_size']
        self._generate_grid()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        r, c = self.agent_pos
        nr, nc = r, c

        if action == MOVE_UP:
            nr = r - 1
        elif action == MOVE_DOWN:
            nr = r + 1
        elif action == MOVE_LEFT:
            nc = c - 1
        elif action == MOVE_RIGHT:
            nc = c + 1
        elif action == INTERACT:
            self._interact(r, c)

        if action in (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT):
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                cell_type = self.grid[nr, nc]
                if cell_type != WALL and cell_type != DOOR_CLOSED:
                    self.agent_pos = [nr, nc]

        terminated = False
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), 0.0, terminated, truncated, {}

    def _interact(self, r, c):
        cell = self.grid[r, c]
        if cell == FOLDER_CLOSED:
            self.grid[r, c] = FOLDER_OPEN
        elif cell == BUTTON_UP:
            self.grid[r, c] = BUTTON_DOWN
        elif cell == DOOR_CLOSED:
            self.grid[r, c] = DOOR_OPEN

    def render(self, mode='ansi'):
        if mode == 'ansi':
            chars = {
                EMPTY: '.', WALL: '#', FOLDER_CLOSED: '[', FOLDER_OPEN: ']',
                BUTTON_UP: '(', BUTTON_DOWN: ')', DOOR_CLOSED: '+', DOOR_OPEN: '-'
            }
            lines = []
            for r in range(self.grid_size):
                line = ''
                for c in range(self.grid_size):
                    if [r, c] == self.agent_pos:
                        line += '@'
                    else:
                        line += chars.get(self.grid[r, c], '?')
                lines.append(line)
            return '\n'.join(lines)
        else:
            return self._render_pixels(self.cell_size)

    def get_interaction_count(self):
        opened = np.sum(self.grid == FOLDER_OPEN)
        pressed = np.sum(self.grid == BUTTON_DOWN)
        opened_doors = np.sum(self.grid == DOOR_OPEN)
        return int(opened + pressed + opened_doors)
