import numpy as np

# Ornstein-Uhlenbeck.
# Taken from https://gym.openai.com/evaluations/eval_y44gvOLNRqckK38LtsP1Q
class OUNoise:
    def __init__(self, actionDim, mu = 0.0, theta = 0.15, sigma = 0.3, seed = 123):
        self.actionDim, self.mu, self.theta, self.sigma = actionDim, mu, theta, sigma
        self.state = np.ones(self.actionDim) * self.mu
        np.random.seed(seed)
        self.reset()

    # Reset process.
    def reset(self):
        self.state = np.ones(self.actionDim) * self.mu

    # Compute noise and
    # update process state.
    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
