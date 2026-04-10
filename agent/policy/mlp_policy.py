import numpy as np
from agent.policy.base_policy import Policy


class MLPPolicy(Policy):
    PARAM_MIN = -6.0
    PARAM_MAX = 6.0

    def __init__(self, dim_states, dim_actions, hidden_dim=8):
        super().__init__(dim_states, dim_actions)
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.hidden_dim = hidden_dim
        self.w1 = np.zeros((self.dim_states, self.hidden_dim), dtype=np.float32)
        self.b1 = np.zeros((1, self.hidden_dim), dtype=np.float32)
        self.w2 = np.zeros((self.hidden_dim, self.dim_actions), dtype=np.float32)
        self.b2 = np.zeros((1, self.dim_actions), dtype=np.float32)

    @property
    def param_count(self):
        return (
            self.dim_states * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.dim_actions
            + self.dim_actions
        )

    def initialize_policy(self):
        self.w1 = np.round(
            np.random.normal(0.0, 0.2, size=(self.dim_states, self.hidden_dim)), 1
        )
        self.b1 = np.round(np.random.normal(0.0, 0.1, size=(1, self.hidden_dim)), 1)
        self.w2 = np.round(
            np.random.normal(0.0, 0.15, size=(self.hidden_dim, self.dim_actions)), 1
        )
        self.b2 = np.round(np.random.normal(0.0, 0.1, size=(1, self.dim_actions)), 1)
        self.w1 = np.clip(self.w1, self.PARAM_MIN, self.PARAM_MAX)
        self.b1 = np.clip(self.b1, self.PARAM_MIN, self.PARAM_MAX)
        self.w2 = np.clip(self.w2, self.PARAM_MIN, self.PARAM_MAX)
        self.b2 = np.clip(self.b2, self.PARAM_MIN, self.PARAM_MAX)

    def get_action(self, state):
        state = state.T
        hidden = np.tanh(np.matmul(state, self.w1) + self.b1)
        return np.tanh(np.matmul(hidden, self.w2) + self.b2)

    def update_policy(self, flat_parameters):
        if flat_parameters is None:
            return
        flat_parameters = np.array(flat_parameters, dtype=np.float32).reshape(-1)
        flat_parameters = np.clip(flat_parameters, self.PARAM_MIN, self.PARAM_MAX)
        flat_parameters = np.round(flat_parameters, 1)

        cursor = 0
        w1_size = self.dim_states * self.hidden_dim
        b1_size = self.hidden_dim
        w2_size = self.hidden_dim * self.dim_actions
        b2_size = self.dim_actions

        self.w1 = flat_parameters[cursor : cursor + w1_size].reshape(
            self.dim_states, self.hidden_dim
        )
        cursor += w1_size
        self.b1 = flat_parameters[cursor : cursor + b1_size].reshape(1, self.hidden_dim)
        cursor += b1_size
        self.w2 = flat_parameters[cursor : cursor + w2_size].reshape(
            self.hidden_dim, self.dim_actions
        )
        cursor += w2_size
        self.b2 = flat_parameters[cursor : cursor + b2_size].reshape(1, self.dim_actions)

    def get_parameters(self):
        return np.concatenate(
            (
                self.w1.reshape(-1),
                self.b1.reshape(-1),
                self.w2.reshape(-1),
                self.b2.reshape(-1),
            )
        )

    def __str__(self):
        return (
            "W1:\n"
            + "\n".join(", ".join(str(x) for x in row) for row in self.w1)
            + "\nB1:\n"
            + "\n".join(", ".join(str(x) for x in row) for row in self.b1)
            + "\nW2:\n"
            + "\n".join(", ".join(str(x) for x in row) for row in self.w2)
            + "\nB2:\n"
            + "\n".join(", ".join(str(x) for x in row) for row in self.b2)
            + "\n"
        )
