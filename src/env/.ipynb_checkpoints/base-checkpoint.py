import json

class BaseEnv:
    """
    A stub environment example for demonstration.
    In practice, you would implement reset, step,
    and any additional logic for your text environment.
    """
    def __init__(self, game: str = "DefaultGame", seed: int = 0):
        self.game = game
        self.rng = random.Random(seed)
        self.state = None
        self.score = 0

    def reset(self, seed: int = 0):
        self.rng.seed(seed)
        self.state = "Welcome to the custom environment: " + self.game
        self.score = 0
        info = {
            "valid_actions": ["look around", "open door", "inspect item"],
            "custom_score": self.score
        }
        return self.state, info

    def step(self, action_idx: int):
        # For demonstration, pick a random reward for each action.
        reward = self.rng.randint(0, 3)
        self.score += reward

        # We'll randomly end the episode with some probability.
        done = self.rng.random() < 0.1

        self.state = f"State updated after action {action_idx}"
        info = {
            "valid_actions": ["look around", "open door", "inspect item"],
            "custom_score": self.score
        }
        return self.state, reward, done, info

    def curr_points(self):
        # A stand-in for "get current score."
        return self.score