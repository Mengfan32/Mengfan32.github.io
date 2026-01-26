from agent.policy.q_table import QTable
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.base_world import BaseWorld
import traceback
import numpy as np
import re
import time
from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.replay_buffer import QTableRewardTrajBuffer


class LLMNumOptimQTableSemanticsAgent:
    def __init__(
        self,
        logdir,
        actions,
        states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        optimum,
        env_kwargs=None,
        env_desc_file=None,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.actions = actions
        self.states = states
        self.optimum = optimum
        self.env_kwargs = env_kwargs
        self.env_desc_file = env_desc_file

        self.q_table = QTable(actions=actions, states=states)
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        self.traj_buffer = ReplayBuffer(max_traj_count, max_traj_length)
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.rank = len(self.q_table.mapping)

    def rollout_episode(self, world: BaseWorld, logging_file, record=True):
        state = world.reset()
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        if record:
            self.traj_buffer.start_new_trajectory()
        while not done:
            action = self.q_table.get_action(state)
            # Robust conversion: handle scalar, list/tuple, and numpy array
            try:
                arr = np.asarray(action)
                if arr.ndim == 0:
                    action = int(arr.item())
                else:
                    action = int(arr.reshape(-1)[0])
            except Exception:
                # Fallback: try direct int conversion
                action = int(action)
            next_state, reward, done = world.step(action)
            logging_file.write(f"{state} | {action} | {reward}\n")
            if record:
                self.traj_buffer.add_step(state, action, reward)
            state = next_state
            step_idx += 1
            self.total_steps += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        return world.get_accu_reward()

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.q_table.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=True)
            self.replay_buffer.add(
                np.array(
                    [self.q_table.mapping[i] for i in range(len(self.q_table.mapping))]
                ),
                result,
            )
            logging_file.close()
            print(f"Result: {result}")

    def train_policy(self, world: BaseWorld, logdir):

        def parse_parameters(input_text):
            # Robust parsing for LLM responses. First try to find explicit "params[i]: value"
            # across the entire response. If that fails or is incomplete, fall back to
            # extracting all numeric values and using the first `rank` of them.
            s = input_text
            print("response:", s)
            indexed_pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
            indexed_matches = indexed_pattern.findall(s)

            # If we have indexed matches, place them into the correct order
            if indexed_matches:
                temp = [None] * self.rank
                for idx_str, val_str in indexed_matches:
                    try:
                        idx = int(idx_str)
                        if 0 <= idx < self.rank:
                            temp[idx] = float(val_str)
                    except Exception:
                        continue
                # If any entries are still None, try to fill from plain numeric extraction
                if any(x is None for x in temp):
                    float_pattern = re.compile(r"[+-]?\d+(?:\.\d+)?")
                    float_matches = float_pattern.findall(s)
                    # Use float matches in order to fill empty slots
                    fm_iter = iter([float(x) for x in float_matches])
                    for i in range(self.rank):
                        if temp[i] is None:
                            try:
                                temp[i] = next(fm_iter)
                            except StopIteration:
                                break
                if all(x is not None for x in temp):
                    return np.array(temp).reshape((self.rank,))

            # Fallback: extract the first `rank` numeric tokens from the whole text
            float_pattern = re.compile(r"[+-]?\d+(?:\.\d+)?")
            float_matches = float_pattern.findall(s)
            results = [float(x) for x in float_matches]
            if len(results) >= self.rank:
                return np.array(results[: self.rank]).reshape((self.rank,))

            # If we still don't have enough numbers, raise an informative error
            raise ValueError(
                f"Could not parse {self.rank} parameters from LLM output. Found {len(results)} numbers.\nFull response:\n{s}"
            )

        def str_nd_examples(replay_buffer: EpisodeRewardBufferNoBias, traj_buffer: ReplayBuffer, n):

            all_parameters = []
            for weights, reward in replay_buffer.buffer:
                parameters = weights
                all_parameters.append((parameters.reshape(-1), reward))

            text = ""
            print('Num trajs in buffer:', len(traj_buffer.buffer))
            print('Num params in buffer:', len(all_parameters))
            for idx, (parameters, reward) in enumerate(all_parameters):
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {parameters[i]:.5g}; "
                fxy = reward
                l += f"f(params): {fxy:.2f}\n"
                l += f"Trajectory: {traj_buffer.buffer[idx]}\n\n"
                text += l
            return text

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        try:
            new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim_semantics(
                str_nd_examples(self.replay_buffer, self.traj_buffer, self.rank),
                parse_parameters,
                self.training_episodes,
                self.env_desc_file,
                self.rank,
                self.optimum,
                actions=self.actions,
            )
            self.api_call_time += api_time
        except Exception as e:
            print("Exception occurred during policy update")
            print(traceback.format_exc())
            raise e

        print(len(self.q_table.mapping))
        print(new_parameter_list.shape)
        self.q_table.update_policy(new_parameter_list)
        print(len(self.q_table.mapping))
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.q_table.mapping))
        logging_q_file.close()
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        q_reasoning_file = open(q_reasoning_filename, "w")
        q_reasoning_file.write(reasoning)
        q_reasoning_file.close()
        print("Policy updated!")

        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        results = []
        for idx in range(self.num_evaluation_episodes):
            if idx == 0:
                result = self.rollout_episode(world, logging_file, record=True)
            else:
                result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        print(f"Results: {results}")
        result = np.mean(results)
        self.replay_buffer.add(
            np.array(
                [self.q_table.mapping[i] for i in range(len(self.q_table.mapping))]
            ),
            result,
        )

        self.training_episodes += 1

        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = result
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward
    
    def evaluate_policy(self, world: BaseWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        return results
