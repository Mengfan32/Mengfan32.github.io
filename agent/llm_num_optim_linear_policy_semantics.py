from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.linear_policy import LinearPolicy
from agent.policy.mlp_policy import MLPPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.base_world import BaseWorld
import numpy as np
import re
import time
import os
from collections import deque


class LLMNumOptimSemanticAgent:
    PARAM_MIN = -6.0
    PARAM_MAX = 6.0

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
        env_desc_file=None,
        policy_type="linear",
        mlp_hidden_dim=8,
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
        self.env_desc_file = env_desc_file
        self.policy_type = policy_type
        self.mlp_hidden_dim = mlp_hidden_dim
        self.phase_period = 32
        self.policy_update_mix = 0.35 if policy_type == "mlp" else 0.65
        self.recent_param_lookback = 12
        self.min_param_linf_change = 0.3
        self.min_param_l2_change = 1.5
        self.novelty_retry_scales = (1.0, 1.5, 2.0, 3.0)
        self.curriculum_positive_target = 3
        self.success_anchor_update_mix = 0.30
        self.success_anchor_prompt_examples = 8

        if self.policy_type == "mlp":
            if not self.bias:
                raise ValueError("MLP policy currently requires bias=True.")
            param_count = (
                dim_state * mlp_hidden_dim
                + mlp_hidden_dim
                + mlp_hidden_dim * dim_action
                + dim_action
            )
        elif not self.bias:
            param_count = dim_action * dim_state
        else:
            param_count = dim_action * dim_state + dim_action
        self.rank = param_count

        if self.policy_type == "mlp":
            self.policy = MLPPolicy(
                dim_states=dim_state,
                dim_actions=dim_action,
                hidden_dim=mlp_hidden_dim,
            )
        elif not self.bias:
            self.policy = LinearPolicyNoBias(
                dim_actions=dim_action, dim_states=dim_state
            )
        else:
            self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        self.traj_buffer = ReplayBuffer(max_traj_count, max_traj_length)
        self.replay_metadata = deque(maxlen=max_traj_count)
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0

        # Expected observation dimension for policy input (bias is handled inside policy).
        self.policy_input_dim = dim_state

    def _is_walker2d_setup(self):
        desc = (self.env_desc_file or "").lower()
        return "walker2d" in desc and self.dim_action == 6 and self.policy_input_dim >= 19

    def _make_walker2d_linear_warmstart(self, variant_idx):
        if self.policy_type != "linear" or not self.bias or not self._is_walker2d_setup():
            self.policy.initialize_policy()
            return self._sanitize_parameters(self.policy.get_parameters().reshape(-1))

        weights = np.zeros((self.policy_input_dim, self.dim_action), dtype=np.float32)
        bias = np.zeros((1, self.dim_action), dtype=np.float32)

        hip_phase = -(1.4 + 0.15 * (variant_idx % 4))
        knee_phase = 1.0 + 0.1 * (variant_idx % 3)
        ankle_phase = 0.5 + 0.05 * (variant_idx % 5)
        cos_mix = -(0.25 + 0.05 * (variant_idx % 2))
        torso_gain = 1.2 + 0.1 * (variant_idx % 3)
        damping_gain = 0.35 + 0.05 * (variant_idx % 4)
        stance_bias = -0.18 if variant_idx % 2 == 0 else 0.18

        # Torso stabilization: resist torso pitch and excessive vertical motion.
        weights[1] = np.array(
            [-torso_gain, -0.3, -0.15, -torso_gain, -0.3, -0.15], dtype=np.float32
        )
        weights[9] = np.array(
            [-0.25, -0.2, -0.1, -0.25, -0.2, -0.1], dtype=np.float32
        )
        weights[10] = np.array(
            [-0.45, -0.25, -0.1, -0.45, -0.25, -0.1], dtype=np.float32
        )

        # Joint-angle centering and damping keep the gait bounded.
        weights[2] = np.array([-0.7, 0.0, 0.0, 0.15, 0.0, 0.0], dtype=np.float32)
        weights[3] = np.array([0.0, -0.7, 0.0, 0.0, 0.1, 0.0], dtype=np.float32)
        weights[4] = np.array([0.0, 0.0, -0.45, 0.0, 0.0, 0.05], dtype=np.float32)
        weights[5] = np.array([0.15, 0.0, 0.0, -0.7, 0.0, 0.0], dtype=np.float32)
        weights[6] = np.array([0.0, 0.1, 0.0, 0.0, -0.7, 0.0], dtype=np.float32)
        weights[7] = np.array([0.0, 0.0, 0.05, 0.0, 0.0, -0.45], dtype=np.float32)
        weights[11] = np.array(
            [-damping_gain, 0.0, 0.0, 0.08, 0.0, 0.0], dtype=np.float32
        )
        weights[12] = np.array(
            [0.0, -damping_gain, 0.0, 0.0, 0.08, 0.0], dtype=np.float32
        )
        weights[13] = np.array(
            [0.0, 0.0, -0.22, 0.0, 0.0, 0.03], dtype=np.float32
        )
        weights[14] = np.array(
            [0.08, 0.0, 0.0, -damping_gain, 0.0, 0.0], dtype=np.float32
        )
        weights[15] = np.array(
            [0.0, 0.08, 0.0, 0.0, -damping_gain, 0.0], dtype=np.float32
        )
        weights[16] = np.array(
            [0.0, 0.0, 0.03, 0.0, 0.0, -0.22], dtype=np.float32
        )

        # Forward-velocity shaping: encourage continued push once moving right.
        weights[8] = np.array([-0.22, 0.18, 0.06, -0.22, 0.18, 0.06], dtype=np.float32)

        # Phase features create an alternating left/right gait.
        weights[17] = np.array(
            [hip_phase, knee_phase, ankle_phase, -hip_phase, -knee_phase, -ankle_phase],
            dtype=np.float32,
        )
        weights[18] = np.array(
            [cos_mix, -0.35, -0.12, -cos_mix, 0.35, 0.12], dtype=np.float32
        )

        bias[0] = np.array(
            [stance_bias, -0.3, 0.1, -stance_bias, -0.3, 0.1], dtype=np.float32
        )

        params = np.concatenate((weights, bias), axis=0).reshape(-1)
        perturbation = np.random.normal(0.0, 0.12, size=params.shape).astype(np.float32)
        phase_boost = np.zeros_like(params)
        phase_start = 17 * self.dim_action
        phase_end = 19 * self.dim_action
        phase_boost[phase_start:phase_end] = np.random.normal(
            0.0, 0.08, size=phase_end - phase_start
        ).astype(np.float32)
        params = params + perturbation + phase_boost
        return self._sanitize_parameters(params)

    def _load_parameters_from_episode_file(self, filename):
        with open(filename, "r") as f:
            text = f.read()
        values = [
            float(x)
            for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        ]
        if len(values) < self.rank:
            return None
        return np.array(values[: self.rank]).reshape(-1)

    def _load_rollout_summary_from_file(self, filename):
        with open(filename, "r") as f:
            text = f.read()
        rewards = [
            float(x)
            for x in re.findall(r"Total reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
        ]
        summary_matches = re.findall(
            r"Summary:\s*alive_steps=(\d+),\s*mean_x_velocity=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*forward_ratio=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*forward_displacement=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*walking_score=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            text,
        )
        if len(rewards) == 0:
            return None, None
        metadata = None
        if summary_matches:
            alive_steps = [int(x[0]) for x in summary_matches]
            mean_x_velocity = [float(x[1]) for x in summary_matches]
            forward_ratio = [float(x[2]) for x in summary_matches]
            forward_displacement = [float(x[3]) for x in summary_matches]
            walking_score = [float(x[4]) for x in summary_matches]
            metadata = {
                "alive_steps": float(np.mean(alive_steps)),
                "mean_x_velocity": float(np.mean(mean_x_velocity)),
                "forward_ratio": float(np.mean(forward_ratio)),
                "forward_displacement": float(np.mean(forward_displacement)),
                "walking_score": float(np.mean(walking_score)),
            }
        return float(np.mean(rewards)), metadata

    def _ensure_replay_metadata_length(self):
        while len(self.replay_metadata) < len(self.replay_buffer.buffer):
            self.replay_metadata.append({})

    def _compute_walking_score(
        self, alive_steps, mean_x_velocity, forward_ratio, forward_displacement
    ):
        # Two-stage curriculum:
        # stage 1 finds any positive forward motion; stage 2 pushes for more
        # stable walking once multiple positive examples exist.
        if self._count_positive_examples() >= self.curriculum_positive_target:
            success_disp = 0.2
            success_vx = 0.1
        else:
            success_disp = 0.0
            success_vx = 0.0

        if forward_displacement <= success_disp or mean_x_velocity < success_vx:
            backward_penalty = min(forward_displacement, 0.0) * 55.0
            return (
                -100.0
                + forward_displacement * 40.0
                + mean_x_velocity * 20.0
                + backward_penalty
            )
        return (
            forward_displacement * 90.0
            + mean_x_velocity * 80.0
            + forward_ratio * 25.0
            + min(alive_steps, 250) * 0.1
        )

    def _count_positive_examples(self):
        return sum(
            1
            for metadata in self.replay_metadata
            if metadata and metadata.get("forward_displacement", 0.0) > 0.0
        )

    def _collect_parameter_metadata_pairs(self):
        self._ensure_replay_metadata_length()
        pairs = []
        for idx, (weights, reward) in enumerate(self.replay_buffer.buffer):
            metadata = self.replay_metadata[idx] if idx < len(self.replay_metadata) else {}
            pairs.append((weights.reshape(-1), reward, metadata))
        return pairs

    def _get_success_examples(self):
        return [
            item
            for item in self._collect_parameter_metadata_pairs()
            if self._is_positive_forward_example(item[2])
        ]

    def _get_best_forward_anchor(self):
        success_examples = self._get_success_examples()
        if not success_examples:
            return None
        return max(
            success_examples,
            key=lambda item: (
                item[2].get("forward_displacement", float("-inf")),
                item[2].get("mean_x_velocity", float("-inf")),
                item[2].get("forward_ratio", float("-inf")),
                item[2].get("alive_steps", float("-inf")),
                item[2].get("walking_score", item[1]),
            ),
        )

    def _is_positive_forward_example(self, metadata):
        if not metadata:
            return False
        try:
            forward_displacement = float(metadata.get("forward_displacement", 0.0))
            mean_x_velocity = float(metadata.get("mean_x_velocity", 0.0))
            forward_ratio = float(metadata.get("forward_ratio", 0.0))
            alive_steps = float(metadata.get("alive_steps", 0.0))
        except Exception:
            return False

        if self._count_positive_examples() >= self.curriculum_positive_target:
            return (
                forward_displacement > 0.2
                and mean_x_velocity > 0.10
                and forward_ratio > 0.50
                and alive_steps >= 80.0
            )
        return forward_displacement > 0.0 and mean_x_velocity > 0.0

    def _get_forward_position(self, world):
        try:
            env = world.env.unwrapped
            data = getattr(env, "data", None)
            qpos = getattr(data, "qpos", None)
            if qpos is not None and len(qpos) > 0:
                return float(qpos[0])
        except Exception:
            pass
        return 0.0

    def _sanitize_parameters(self, params):
        params = np.array(params, dtype=np.float32).reshape(-1)
        params = np.clip(params, self.PARAM_MIN, self.PARAM_MAX)
        params = np.round(params, 1)
        return params

    def _get_recent_parameter_history(self):
        recent = []
        for item in list(self.replay_buffer.buffer)[-self.recent_param_lookback :]:
            params = item[0]
            recent.append(self._sanitize_parameters(params))
        return recent

    def _parameter_distance(self, a, b):
        diff = self._sanitize_parameters(a) - self._sanitize_parameters(b)
        return float(np.max(np.abs(diff))), float(np.linalg.norm(diff))

    def _is_parameter_too_close(self, candidate, references):
        candidate = self._sanitize_parameters(candidate)
        for ref in references:
            linf, l2 = self._parameter_distance(candidate, ref)
            if linf < self.min_param_linf_change or l2 < self.min_param_l2_change:
                return True
        return False

    def _generate_novel_candidate(self, anchor, scale_multiplier):
        anchor = self._sanitize_parameters(anchor)
        base_step = max(self.search_step_size, self.min_param_linf_change) * scale_multiplier
        indices = np.arange(anchor.shape[0], dtype=np.float32)
        pattern = np.where(
            ((indices + self.training_episodes) % 4) < 2,
            1.0,
            -1.0,
        )
        phase_wave = np.sin((indices + 1.0) * (self.training_episodes + 1.0) * 0.37)
        perturbation = pattern * base_step + np.round(phase_wave, 1) * 0.1
        return self._sanitize_parameters(anchor + perturbation)

    def _enforce_parameter_novelty(self, candidate, references, anchor):
        all_refs = [self._sanitize_parameters(anchor)] + [
            self._sanitize_parameters(ref) for ref in references
        ]
        candidate = self._sanitize_parameters(candidate)
        if not self._is_parameter_too_close(candidate, all_refs):
            return candidate

        for scale in self.novelty_retry_scales:
            shifted = self._generate_novel_candidate(anchor, scale)
            if not self._is_parameter_too_close(shifted, all_refs):
                return shifted

        # Last resort: return the furthest shifted candidate even if constraints stay tight.
        return self._generate_novel_candidate(anchor, self.novelty_retry_scales[-1] * 1.5)

    def _augment_state(self, state, step_idx):
        state_arr = np.asarray(state).reshape(-1)
        if state_arr.shape[0] == self.policy_input_dim:
            return state_arr
        if state_arr.shape[0] + 2 == self.policy_input_dim:
            phase = 2.0 * np.pi * (step_idx % self.phase_period) / self.phase_period
            return np.concatenate(
                (
                    state_arr,
                    np.array([np.sin(phase), np.cos(phase)], dtype=state_arr.dtype),
                )
            )
        raise ValueError(
            f"State dimension mismatch: got {state_arr.shape[0]}, expected {self.policy_input_dim} or {self.policy_input_dim - 2}"
        )

    def resume_from_logdir(self, logdir, start_episode):
        resumed_count = 0
        for episode in range(start_episode):
            episode_dir = os.path.join(logdir, f"episode_{episode}")
            params_file = os.path.join(episode_dir, "parameters.txt")
            rollout_file = os.path.join(episode_dir, "training_rollout.txt")
            if not (os.path.exists(params_file) and os.path.exists(rollout_file)):
                continue

            params = self._load_parameters_from_episode_file(params_file)
            mean_reward, metadata = self._load_rollout_summary_from_file(rollout_file)
            if params is None or mean_reward is None:
                continue
            meta = metadata or {}
            meta.setdefault("reward_mean", mean_reward)
            meta.setdefault("score", meta.get("walking_score", mean_reward))
            self.replay_buffer.add(params, meta["score"])
            self.replay_metadata.append(meta)
            resumed_count += 1

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
        raw_state = world.reset()
        start_x = self._get_forward_position(world)
        logging_file.write(
            f"{', '.join([str(x) for x in self.policy.get_parameters().reshape(-1)])}\n"
        )
        logging_file.write(f"parameter ends\n\n")
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        x_velocity_sum = 0.0
        positive_x_velocity_steps = 0
        if record:
            self.traj_buffer.start_new_trajectory()
        while not done:
            flat_state = np.asarray(raw_state).reshape(-1)
            policy_state = self._augment_state(raw_state, step_idx)
            state = np.expand_dims(policy_state, axis=0)
            x_velocity = float(flat_state[8]) if flat_state.shape[0] > 8 else 0.0
            x_velocity_sum += x_velocity
            if x_velocity > 0:
                positive_x_velocity_steps += 1
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            if world.discretize:
                action = np.argmax(action)
                action = np.array([action])
            next_state, reward, done = world.step(action)
            logging_file.write(f"{policy_state} | {action[0]} | {reward}\n")
            if record:
                self.traj_buffer.add_step(state, action, reward)
            raw_state = next_state
            step_idx += 1
            self.total_steps += 1
        mean_x_velocity = x_velocity_sum / max(step_idx, 1)
        forward_ratio = positive_x_velocity_steps / max(step_idx, 1)
        end_x = self._get_forward_position(world)
        forward_displacement = end_x - start_x
        walking_score = self._compute_walking_score(
            step_idx, mean_x_velocity, forward_ratio, forward_displacement
        )
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        logging_file.write(
            "Summary: "
            f"alive_steps={step_idx}, "
            f"mean_x_velocity={mean_x_velocity:.4f}, "
            f"forward_ratio={forward_ratio:.4f}, "
            f"forward_displacement={forward_displacement:.4f}, "
            f"walking_score={walking_score:.4f}\n"
        )
        self.total_episodes += 1
        return {
            "reward": world.get_accu_reward(),
            "alive_steps": step_idx,
            "mean_x_velocity": mean_x_velocity,
            "forward_ratio": forward_ratio,
            "forward_displacement": forward_displacement,
            "walking_score": walking_score,
        }

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            if self._is_walker2d_setup() and self.policy_type == "linear":
                warmstart_params = self._make_walker2d_linear_warmstart(episode)
                self.policy.update_policy(warmstart_params)
            else:
                self.policy.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file)
            self.replay_buffer.add(
                np.array(self.policy.get_parameters()).reshape(-1), result["walking_score"]
            )
            self.replay_metadata.append(
                {
                    "reward_mean": result["reward"],
                    "score": result["walking_score"],
                    "alive_steps": result["alive_steps"],
                    "mean_x_velocity": result["mean_x_velocity"],
                    "forward_ratio": result["forward_ratio"],
                    "forward_displacement": result["forward_displacement"],
                    "walking_score": result["walking_score"],
                }
            )
            logging_file.close()
            print(f"Result: {result}")
        # self.replay_buffer.sort()

    def train_policy(self, world: BaseWorld, logdir):

        def parse_parameters(input_text):
            s = input_text
            print("response:", s)
            pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
            matches = pattern.findall(s)

            parsed = {}
            for idx_str, val_str in matches:
                idx = int(idx_str)
                if 0 <= idx < self.rank:
                    parsed[idx] = float(val_str)

            current = self._sanitize_parameters(
                self.policy.get_parameters().reshape(-1).astype(float)
            )
            results = current.copy()
            for idx, val in parsed.items():
                results[idx] = val

            if len(parsed) != self.rank:
                missing = self.rank - len(parsed)
                print(
                    f"[warn] parsed {len(parsed)}/{self.rank} params from LLM output; "
                    f"filled {missing} missing params from previous policy."
                )

            return self._sanitize_parameters(results)

        def str_nd_examples(replay_buffer: EpisodeRewardBufferNoBias, traj_buffer: ReplayBuffer, n):
            all_parameters = self._collect_parameter_metadata_pairs()

            # Keep prompt size bounded. MLP policies have far more parameters, so they
            # need a much smaller history window to stay within context limits.
            if self.policy_type == "mlp":
                max_examples = 3
                recent_keep = 3
                success_examples = [
                    item for item in all_parameters if self._is_positive_forward_example(item[2])
                ]
                positive_examples = [
                    item
                    for item in all_parameters
                    if item[2].get("forward_displacement", float("-inf")) > 0.0
                ]
                if success_examples:
                    all_parameters = success_examples[-max_examples:]
                elif positive_examples:
                    all_parameters = positive_examples[-max_examples:]
                else:
                    all_parameters = all_parameters[-recent_keep:]
            else:
                max_examples = 32
                recent_keep = 10
                success_examples = [
                    item for item in all_parameters if self._is_positive_forward_example(item[2])
                ]
                if success_examples:
                    success_examples = sorted(
                        success_examples,
                        key=lambda item: (
                            item[2].get("forward_displacement", float("-inf")),
                            item[2].get("mean_x_velocity", float("-inf")),
                            item[2].get("forward_ratio", float("-inf")),
                            item[2].get("alive_steps", float("-inf")),
                        ),
                        reverse=True,
                    )[: self.success_anchor_prompt_examples]
                    recent_examples = all_parameters[-recent_keep:]
                    all_parameters = success_examples + recent_examples
            if len(all_parameters) > max_examples:
                top_keep = max_examples - recent_keep

                recent_indices = list(range(len(all_parameters) - recent_keep, len(all_parameters)))
                remaining_indices = list(range(len(all_parameters) - recent_keep))
                top_indices = sorted(
                    remaining_indices,
                    key=lambda i: all_parameters[i][1],
                    reverse=True,
                )[:top_keep]

                selected_indices = sorted(set(top_indices + recent_indices))
                all_parameters = [all_parameters[i] for i in selected_indices]

            text = ""
            print('Num trajs in buffer:', len(traj_buffer.buffer))
            print('Num params in buffer:', len(all_parameters))
            for idx, (parameters, reward, metadata) in enumerate(all_parameters):
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {parameters[i]:.1f}; "
                score = metadata.get("score", reward)
                reward_mean = metadata.get("reward_mean", reward)
                l += f"walking_score(params): {score:.2f}\n"
                l += f"reward_mean(params): {reward_mean:.2f}\n"
                if metadata:
                    l += (
                        "Locomotion metrics: "
                        f"mean_x_velocity={metadata.get('mean_x_velocity', 0.0):.4f}; "
                        f"forward_ratio={metadata.get('forward_ratio', 0.0):.4f}; "
                        f"forward_displacement={metadata.get('forward_displacement', 0.0):.4f}; "
                        f"alive_steps={metadata.get('alive_steps', 0.0):.1f}; "
                        f"walking_score={metadata.get('walking_score', 0.0):.4f}\n"
                    )
                text += l
            return text

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        positive_example_count = self._count_positive_examples()
        success_anchor = self._get_best_forward_anchor()
        if success_anchor is not None:
            effective_search_step_size = min(self.search_step_size, 0.05)
            curriculum_stage = 3
        elif positive_example_count >= self.curriculum_positive_target:
            effective_search_step_size = min(self.search_step_size, 0.06)
            curriculum_stage = 2
        elif positive_example_count > 0:
            effective_search_step_size = min(self.search_step_size, 0.10)
            curriculum_stage = 1
        else:
            effective_search_step_size = self.search_step_size
            curriculum_stage = 1
        print(
            f"Curriculum stage: {curriculum_stage} "
            f"(positive_examples={positive_example_count}, "
            f"effective_search_step={effective_search_step_size})"
        )
        new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim_semantics(
            str_nd_examples(self.replay_buffer, self.traj_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.env_desc_file,
            self.rank,
            self.optimum,
            effective_search_step_size,
            policy_type=self.policy_type,
            mlp_hidden_dim=self.mlp_hidden_dim,
        )
        self.api_call_time += api_time

        print(self.policy.get_parameters().shape)
        print(new_parameter_list.shape)
        old_parameters = self._sanitize_parameters(self.policy.get_parameters().reshape(-1))
        base_parameters = old_parameters
        active_update_mix = self.policy_update_mix
        if success_anchor is not None:
            base_parameters = self._sanitize_parameters(success_anchor[0])
            active_update_mix = min(active_update_mix, self.success_anchor_update_mix)
            print(
                "Using success anchor: "
                f"forward_displacement={success_anchor[2].get('forward_displacement', 0.0):.4f}, "
                f"mean_x_velocity={success_anchor[2].get('mean_x_velocity', 0.0):.4f}, "
                f"forward_ratio={success_anchor[2].get('forward_ratio', 0.0):.4f}, "
                f"alive_steps={success_anchor[2].get('alive_steps', 0.0):.1f}"
            )
        recent_parameters = self._get_recent_parameter_history()
        new_parameter_list = self._enforce_parameter_novelty(
            new_parameter_list.reshape(-1), recent_parameters, base_parameters
        )
        applied_parameter_list = (
            (1.0 - active_update_mix) * base_parameters
            + active_update_mix * new_parameter_list.reshape(-1)
        )
        applied_parameter_list = self._sanitize_parameters(applied_parameter_list)
        applied_parameter_list = self._enforce_parameter_novelty(
            applied_parameter_list, recent_parameters, base_parameters
        )
        self.policy.update_policy(applied_parameter_list)
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
        rollout_summaries = []
        for idx in range(self.num_evaluation_episodes):
            if idx == 0:
                result = self.rollout_episode(world, logging_file, record=True)
            else:
                result = self.rollout_episode(world, logging_file, record=False)
            rollout_summaries.append(result)
        rewards = [x["reward"] for x in rollout_summaries]
        mean_x_velocity = [x["mean_x_velocity"] for x in rollout_summaries]
        forward_ratio = [x["forward_ratio"] for x in rollout_summaries]
        forward_displacement = [x["forward_displacement"] for x in rollout_summaries]
        alive_steps = [x["alive_steps"] for x in rollout_summaries]
        walking_score = [x["walking_score"] for x in rollout_summaries]
        print(f"Results: {rollout_summaries}")
        result = float(np.mean(rewards))
        metadata = {
            "alive_steps": float(np.mean(alive_steps)),
            "mean_x_velocity": float(np.mean(mean_x_velocity)),
            "forward_ratio": float(np.mean(forward_ratio)),
            "forward_displacement": float(np.mean(forward_displacement)),
            "walking_score": float(np.mean(walking_score)),
        }
        metadata["reward_mean"] = result
        metadata["score"] = metadata["walking_score"]
        self.replay_buffer.add(applied_parameter_list, metadata["score"])
        self.replay_metadata.append(metadata)
        # self.replay_buffer.sort()

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
