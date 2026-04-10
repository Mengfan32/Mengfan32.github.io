import numpy as np
from agent.policy.base_policy import Policy


class LinearPolicy(Policy):
    PARAM_MIN = -6.0
    PARAM_MAX = 6.0

    def __init__(self, dim_states, dim_actions):
        super().__init__(dim_states, dim_actions)

        self.dim_states =dim_states
        self.dim_actions = dim_actions

        self.weight = np.random.rand(self.dim_states, self.dim_actions)
        self.bias = np.random.rand(1, self.dim_actions)

    def initialize_policy(self):
        self.weight = np.round(
            np.random.normal(0.0, 0.4, size=(self.dim_states, self.dim_actions)), 1
        )
        self.bias = np.round(
            np.random.normal(0.0, 0.4, size=(1, self.dim_actions)), 1
        )
        self.weight = np.clip(self.weight, self.PARAM_MIN, self.PARAM_MAX)
        self.bias = np.clip(self.bias, self.PARAM_MIN, self.PARAM_MAX)

    def get_action(self, state):
        state = state.T
        # print(state.shape, self.weight.shape, self.bias.shape)
        # print(np.matmul(state, self.weight).shape, (np.matmul(state, self.weight) + self.bias).shape)
        # print((np.matmul(state, self.weight) + self.bias).shape)
        # print()
        # return np.matmul(state, self.weight + np.array([[2], [1]])) + self.bias + np.array([[-1]])
        return np.matmul(state, self.weight) + self.bias

    def __str__(self):
        output = "Weights:\n"
        for w in self.weight:
            output += ", ".join([str(i) for i in w])
            output += "\n"

        output += "Bias:\n"
        for b in self.bias:
            output += ", ".join([str(i) for i in b])
            output += "\n"

        return output

    def update_policy(self, weight_and_bias_list):
        if weight_and_bias_list is None:
            return

        weight_and_bias_list = np.array(weight_and_bias_list, dtype=np.float32)
        weight_and_bias_list = np.clip(weight_and_bias_list, self.PARAM_MIN, self.PARAM_MAX)
        weight_and_bias_list = np.round(weight_and_bias_list, 1)
        weight_and_bias_list = weight_and_bias_list.reshape(
            self.dim_states + 1, self.dim_actions
        )
        self.weight = np.array(weight_and_bias_list[:-1])
        self.bias = np.expand_dims(np.array(weight_and_bias_list[-1]), axis=0)
    
    def get_parameters(self):
        parameters = np.concatenate((self.weight, self.bias), axis=0)
        return parameters
