from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.base_world import BaseWorld
import numpy as np
import re
import time
import os


class LLMNumOptimAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        bias,
        optimum,
        search_step_size,
        param_update_step_limit=0.2,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.bias = bias
        self.optimum = optimum
        self.search_step_size = search_step_size
        self.param_update_step_limit = param_update_step_limit

        if not self.bias:
            param_count = dim_action * dim_state
        else:
            param_count = dim_action * dim_state + dim_action
        self.rank = param_count

        if not self.bias:
            self.policy = LinearPolicyNoBias(
                dim_actions=dim_action, dim_states=dim_state
            )
        else:
            self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0

        if self.bias:
            self.dim_state += 1

    def _load_parameters_from_episode_file(self, filename):
        with open(filename, "r") as f:
            text = f.read()
        # Parse all numeric values from "Weights"/"Bias" dump.
        values = [
            float(x)
            for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        ]
        if len(values) < self.rank:
            return None
        return np.array(values[: self.rank]).reshape(-1)

    def _load_mean_reward_from_rollout_file(self, filename):
        with open(filename, "r") as f:
            text = f.read()
        rewards = [
            float(x)
            for x in re.findall(r"Total reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
        ]
        if len(rewards) == 0:
            return None
        return float(np.mean(rewards))

    def resume_from_logdir(self, logdir, start_episode):
        """
        True resume:
        - Rebuild replay buffer from episode_0 ... episode_{start_episode-1}
        - Restore current policy from episode_{start_episode-1}/parameters.txt
        - Restore training episode counter to start_episode
        """
        resumed_count = 0
        for episode in range(start_episode):
            episode_dir = os.path.join(logdir, f"episode_{episode}")
            params_file = os.path.join(episode_dir, "parameters.txt")
            rollout_file = os.path.join(episode_dir, "training_rollout.txt")
            if not (os.path.exists(params_file) and os.path.exists(rollout_file)):
                continue

            params = self._load_parameters_from_episode_file(params_file)
            mean_reward = self._load_mean_reward_from_rollout_file(rollout_file)
            if params is None or mean_reward is None:
                continue
            self.replay_buffer.add(params, mean_reward)
            resumed_count += 1

        # Restore the latest available policy before the resume point.
        for episode in range(start_episode - 1, -1, -1):
            params_file = os.path.join(logdir, f"episode_{episode}", "parameters.txt")
            if not os.path.exists(params_file):
                continue
            params = self._load_parameters_from_episode_file(params_file)
            if params is None:
                continue
            self.policy.update_policy(params)
            break

        self.training_episodes = start_episode
        print(
            f"[resume] loaded {resumed_count} historical episodes into replay buffer; "
            f"set training_episodes={self.training_episodes}"
        )

    def rollout_episode(self, world: BaseWorld, logging_file, record=True):
        state = world.reset()
        state = np.expand_dims(state, axis=0)
        logging_file.write(
            f"{', '.join([str(x) for x in self.policy.get_parameters().reshape(-1)])}\n"
        )
        logging_file.write(f"parameter ends\n\n")
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        while not done:
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            if world.discretize:
                action = np.argmax(action)
                action = np.array([action])
            next_state, reward, done = world.step(action)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")
            state = next_state
            step_idx += 1
            self.total_steps += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        if record:
            self.replay_buffer.add(
                self.policy.get_parameters(), world.get_accu_reward()
            )
        return world.get_accu_reward()

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.policy.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file)
            print(f"Result: {result}")

    def train_policy(self, world: BaseWorld, logdir):

        def parse_parameters(input_text):
            # Parse params across the whole response (LLM may wrap lines).
            preview = input_text.split("\n")[0]
            print("response:", preview)
            pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
            matches = pattern.findall(input_text)

            # Build an index->value map; if duplicated, keep the latest occurrence.
            parsed = {}
            for idx_str, val_str in matches:
                idx = int(idx_str)
                if 0 <= idx < self.rank:
                    parsed[idx] = float(val_str)

            # Robust fallback: if model output is truncated, keep old values for missing params.
            current = self.policy.get_parameters().reshape(-1).astype(float)
            results = current.copy()
            for idx, val in parsed.items():
                results[idx] = val

            if len(parsed) != self.rank:
                missing = self.rank - len(parsed)
                print(
                    f"[warn] parsed {len(parsed)}/{self.rank} params from LLM output; "
                    f"filled {missing} missing params from previous policy."
                )
            return np.array(results).reshape(-1)

        def str_nd_examples(replay_buffer: EpisodeRewardBufferNoBias, n):
            """
            Build a bounded-size prompt payload to avoid LLM context overflow.
            We keep a diverse subset: recent samples + best + worst.
            """
            all_rows = []
            for idx, (weights, reward) in enumerate(replay_buffer.buffer):
                all_rows.append((idx, weights.reshape(-1), reward))

            if len(all_rows) == 0:
                return ""

            # Keep prompt bounded. These values keep context safely below model limits.
            max_examples = 40
            recent_k = 20
            best_k = 10
            worst_k = 10

            selected = []
            # Recent
            selected.extend(all_rows[-recent_k:])
            # Best / worst by reward
            selected.extend(sorted(all_rows, key=lambda x: x[2], reverse=True)[:best_k])
            selected.extend(sorted(all_rows, key=lambda x: x[2])[:worst_k])

            # Deduplicate while preserving order by idx
            seen = set()
            deduped = []
            for row in selected:
                idx = row[0]
                if idx in seen:
                    continue
                seen.add(idx)
                deduped.append(row)

            # If still too many, keep most recent max_examples
            deduped = deduped[-max_examples:]

            text = ""
            for _, parameters, reward in deduped:
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {parameters[i]:.5g}; "
                l += f"f(params): {reward:.2f}\n"
                text += l
            return text

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim(
            str_nd_examples(self.replay_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.rank,
            self.optimum,
            self.search_step_size
        )
        self.api_call_time += api_time

        print(self.policy.get_parameters().shape)
        print(new_parameter_list.shape)
        # Trust-region style parameter update:
        # limit per-parameter change to avoid unstable large jumps.
        current_parameters = self.policy.get_parameters().reshape(-1).astype(float)
        proposed_parameters = np.array(new_parameter_list).reshape(-1).astype(float)
        if (
            self.param_update_step_limit is not None
            and float(self.param_update_step_limit) > 0
            and len(current_parameters) == len(proposed_parameters)
        ):
            limit = float(self.param_update_step_limit)
            delta = proposed_parameters - current_parameters
            clipped_delta = np.clip(delta, -limit, limit)
            clipped_count = int(np.sum(np.abs(delta) > limit))
            if clipped_count > 0:
                print(
                    f"[step-limit] clipped {clipped_count}/{len(delta)} params "
                    f"with limit={limit}"
                )
            applied_parameters = current_parameters + clipped_delta
        else:
            applied_parameters = proposed_parameters

        self.policy.update_policy(applied_parameters)
        print(self.policy.get_parameters().shape)
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.policy))
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
                result = self.rollout_episode(world, logging_file, record=False)
            else:
                result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        print(f"Results: {results}")
        result = np.mean(results)
        self.replay_buffer.add(applied_parameters, result)

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
