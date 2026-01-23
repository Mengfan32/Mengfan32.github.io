import gymnasium as gym
import random
import numpy as np
import os
import time
from jinja2 import Template
from openai import OpenAI
import google.generativeai as genai

import anthropic
import time


class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_conversation = []
        assert llm_model_name in [
            "o1-preview",
            "gpt-4o",
            "gemini-2.0-flash-exp",
            "gpt-4o-mini",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17",
            "o3-mini-2025-01-31",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "claude-3-7-sonnet-20250219",
            "gemini-2.5-flash",  
            "gemini-2.0-flash",  
            "gemini-2.5-pro",   
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-reasoner",
        ]
        self.llm_model_name = llm_model_name
        if "gemini" in llm_model_name:
            self.model_group = "gemini"
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        elif "claude" in llm_model_name:
            self.model_group = "anthropic"
            self.client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif "deepseek" in llm_model_name:
            self.model_group = "openai"  
            self.client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com"  
            )
        else:
            self.model_group = "openai"
            self.client = OpenAI()

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        if self.model_group == "openai":
            self.llm_conversation.append({"role": role, "content": text})
        elif self.model_group == "anthropic":
            self.llm_conversation.append({"role": role, "content": text})
        else:
            self.llm_conversation.append({"role": role, "parts": text})

    def query_llm(self):
        for attempt in range(10):
            try:
                if self.model_group == "openai":

                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                    )
                    response = completion.choices[0].message.content
                elif self.model_group == "anthropic":
                    message = self.client.messages.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        max_tokens=1024,
                    )
                    response = message.content[0].text
                else:
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    chat_session = model.start_chat(history=self.llm_conversation[:-1])
                    response = chat_session.send_message(
                        self.llm_conversation[-1]["parts"]
                    )
                    response = response.text
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                if attempt == 9:
                    raise Exception("Failed")
                else:
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)
                    continue

            if self.model_group == "openai":
                # add the response to self.llm_conversation
                self.add_llm_conversation(response, "assistant")
            else:
                self.add_llm_conversation(response, "model")

            return response

    def query_llm_multiple_response(self, num_responses, temperature):
        for attempt in range(5):
            try:
                if self.model_group == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        n=num_responses,
                        temperature=temperature,
                    )
                    responses = [
                        completion.choices[i].message.content
                        for i in range(num_responses)
                    ]
                else:
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    responses = model.generate_content(
                        contents=self.llm_conversation,
                        generation_config=genai.GenerationConfig(
                            candidate_count=num_responses,
                            temperature=temperature,
                        ),
                    )
                    responses = [
                        "\n".join([x.text for x in c.content.parts])
                        for c in responses.candidates
                    ]

            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                if attempt == 4:
                    raise Exception("Failed")
                else:
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)

            return responses

    def parse_parameters(self, parameters_string):
        new_parameters_list = []

        # Update the Q-table based on the new Q-table
        for row in parameters_string.split("\n"):
            if row.strip().strip(","):
                try:
                    parameters_row = [
                        float(x.strip().strip(",")) for x in row.split(",")
                    ]
                    new_parameters_list.append(parameters_row)
                except Exception as e:
                    print(e)

        return new_parameters_list

    def llm_update_parameters(self, parameters, replay_buffer, parse_parameters=None):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "replay_buffer_string": str(replay_buffer),
                "parameters_string": str(parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        if self.model_group == "openai":
            self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        else:
            self.add_llm_conversation(new_parameters_with_reasoning, "model")
        self.add_llm_conversation(
            self.llm_output_conversion_template.render(),
            "user",
        )
        new_parameters = self.query_llm()

        if parse_parameters is None:
            new_parameters_list = self.parse_parameters(new_parameters)
        else:
            new_parameters_list = parse_parameters(new_parameters)

        return new_parameters_list, [new_parameters_with_reasoning, new_parameters]

    def llm_update_parameters_sas(self, episode_reward_buffer, parse_parameters=None):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {"episode_reward_buffer_string": str(episode_reward_buffer)}
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        self.add_llm_conversation(
            self.llm_output_conversion_template.render(),
            "user",
        )
        new_parameters = self.query_llm()

        if parse_parameters is None:
            new_parameters_list = self.parse_parameters(new_parameters)
        else:
            new_parameters_list = parse_parameters(new_parameters)

        return new_parameters_list, [
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            new_parameters,
        ]

    def llm_update_parameters_num_optim(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "actions": actions,
            }
        )

        self.add_llm_conversation(system_prompt, "user")

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time

        # print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            api_time,
        )

    def llm_update_parameters_num_optim_q_table(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        actions,
        num_states,
        optimum,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "actions": actions,
                "rank": num_states,
                "optimum": str(optimum),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_update_parameters_num_optim_imitation(
        self,
        demonstrations_str,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "expert_demonstration_string": demonstrations_str,
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_propose_parameters_num_optim_based_on_anchor(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_propose_multiple_parameters_num_optim_based_on_anchor(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
        num_candidates,
        temperature,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        # print(system_prompt)
        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning_list = self.query_llm_multiple_response(
            num_candidates, temperature
        )
        # print(new_parameters_with_reasoning_list)

        new_parameters_list = []
        reasonings_list = []
        for new_params in new_parameters_with_reasoning_list:
            new_params_np = parse_parameters(new_params)
            new_parameters_list.append(new_params_np)
            reasonings_list.append(new_params)

        return (
            system_prompt,
            new_parameters_list,
            reasonings_list,
        )

    def llm_propose_parameters_num_optim_based_on_anchor_thread(
        self,
        new_candidates,
        new_idx,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)
        new_candidates[new_idx] = new_parameters_list

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_update_parameters_num_optim_semantics(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        env_desc_file,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "env_description": env_desc_file,
                "step_number": str(step_number),
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "actions": actions,
            }
        )


        self.add_llm_conversation(system_prompt, "user")

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time

        # print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            api_time,
        )
