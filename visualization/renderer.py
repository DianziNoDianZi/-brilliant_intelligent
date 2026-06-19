import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
import numpy as np
from collections import deque

import config

from environment.grid_world import (
    EMPTY, WALL, FOLDER_CLOSED, FOLDER_OPEN,
    BUTTON_UP, BUTTON_DOWN, DOOR_CLOSED, DOOR_OPEN,
)

CELL_COLORS = {
    EMPTY: '#f0f0f0',
    WALL: '#404040',
    FOLDER_CLOSED: '#b48828',
    FOLDER_OPEN: '#ffc832',
    BUTTON_UP: '#5082c8',
    BUTTON_DOWN: '#2864c8',
    DOOR_CLOSED: '#a06428',
    DOOR_OPEN: '#dcaa50',
}


class Phase1Renderer:
    """Matplotlib-based renderer showing grid world + live metrics curves."""

    def __init__(self, env, config):
        self.env = env
        self.config = config

        plt.ion()
        mode = getattr(config, 'OBS_MODE', 'vector')
        title = f'Phase 1 — {"Visual + Ensemble" if mode == "pixels" else "Vector"} Agent'
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.suptitle(title, fontsize=13)

        # Grid subplot
        self.ax_grid = self.fig.add_axes([0.05, 0.35, 0.55, 0.60])
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        self.grid_patches = []

        # Metrics subplot
        self.ax_metrics = self.fig.add_axes([0.05, 0.05, 0.90, 0.25])
        self.ax_metrics.set_xlabel('Step')
        self.ax_metrics.set_ylabel('Value')
        self.plot_window = 200
        self.steps = deque(maxlen=self.plot_window)
        self.rewards = deque(maxlen=self.plot_window)
        self.errors = deque(maxlen=self.plot_window)
        self.interactions = deque(maxlen=self.plot_window)
        self.step_counter = 0

        self.line_reward, = self.ax_metrics.plot([], [], label='Curiosity Reward',
                                                  color='#2ecc71', lw=1)
        self.line_error, = self.ax_metrics.plot([], [], label='Prediction Error',
                                                color='#e74c3c', lw=1)
        self.line_interact, = self.ax_metrics.plot([], [], label='Interactions',
                                                    color='#3498db', lw=1)
        self.ax_metrics.legend(fontsize=8, loc='upper left')
        self.ax_metrics.grid(True, alpha=0.3)

        self._draw_grid()

    def _draw_grid(self):
        self.ax_grid.clear()
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])

        size = self.env.grid_size
        patches = []
        for r in range(size):
            for c in range(size):
                cell = self.env.grid[r, c]
                color = CELL_COLORS.get(cell, '#cccccc')
                rect = Rectangle((c, size - 1 - r), 1, 1,
                                 facecolor=color, edgecolor='#888', lw=0.5)
                patches.append(rect)

        # Agent
        ar, ac = self.env.agent_pos
        agent = Circle((ac + 0.5, size - 1 - ar + 0.5), 0.35,
                       facecolor='#e74c3c', edgecolor='#c0392b', lw=2, zorder=10)

        collection = PatchCollection(patches, match_original=True, zorder=1)
        self.ax_grid.add_collection(collection)
        self.ax_grid.add_patch(agent)
        self.ax_grid.set_xlim(0, size)
        self.ax_grid.set_ylim(0, size)

    def update(self, env):
        self.env = env
        self.step_counter += 1

        self._draw_grid()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def add_metric(self, reward, error, interactions):
        self.steps.append(self.step_counter)
        self.rewards.append(reward)
        self.errors.append(error)
        self.interactions.append(interactions)

    def update_plot(self, history):
        if len(self.steps) < 2:
            return

        x = list(self.steps)
        self.line_reward.set_data(x, list(self.rewards))
        self.line_error.set_data(x, list(self.errors))
        self.line_interact.set_data(x, list(self.interactions))

        self.ax_metrics.relim()
        self.ax_metrics.autoscale_view()

    def close(self):
        plt.ioff()
        plt.close(self.fig)
