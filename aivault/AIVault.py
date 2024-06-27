import os
import time
import logging
import random
import importlib
import tiktoken
from typing import List, Optional, Dict
from pydantic import BaseModel


class ChatCompletions(BaseModel):
    model: str = "gpt-4o"
    messages: List[dict] = None
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 1.0
    tools: Optional[List[dict]] = None
    tools_choice: Optional[str] = "auto"
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = 4096
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = ""


def get_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def get_providers():
    providers = []
    for file in os.listdir("./aivault/providers"):
        if file.endswith(".py") and not file.startswith("__"):
            module = importlib.import_module(f"aivault.providers.{file[:-3]}")
            for obj in dir(module):
                if obj.endswith("Provider"):
                    providers.append(getattr(module, obj))
    provider_list = []
    for provider in providers:
        provider_list.append(
            {
                "name": provider.__name__,
                "class": provider,
                "models": provider.models,
            }
        )
    return provider_list


class AIVault:
    def __init__(self, **kwargs):
        self.providers = get_providers()
        self.provider = random.choice(self.providers)["class"]()
        self.provider_name = self.provider.__class__.__name__
        self.friendly_provider_name = str(self.provider_name).split("Provider")[0]
        self.model = random.choice(self.provider.models)
        self.failures = []
        self.provider_failure_count = 0

    async def inference(self, prompt, images: list = [], **kwargs):
        logging.info(
            f"[AIVault] Using provider: {self.friendly_provider_name} with model: {self.model}"
        )
        try:
            return await self.provider.inference(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                images=images,
                **kwargs,
            )
        except Exception as e:
            logging.error(f"[AIVault] {e}")
            self.failures.append({"provider": self.provider_name, "model": self.model})
            if len(self.failures) < len(self.providers):
                available_providers = self.get_available_providers()
                if available_providers:
                    provider = random.choice(available_providers)
                    self.provider = provider["class"]()
                    self.provider_name = provider["name"]
                    self.model = random.choice(provider["models"])
                    logging.info(
                        f"[AIVault] Switching to provider: {self.friendly_provider_name} with model: {self.model}"
                    )
                    return await self.inference(prompt=prompt, images=images)
                else:
                    return "No available providers. Unable to retrieve response."
            else:
                # Try all providers 3 times before fully failing.
                self.provider_failure_count += 1
                self.failures = []
                if self.provider_failure_count < 3:
                    return await self.inference(prompt=prompt, images=images)
                return "All providers have failed. Unable to retrieve response."

    def get_available_providers(self):
        available_providers = []
        for provider in self.providers:
            provider_models = provider["models"]
            if not isinstance(provider_models, list):
                provider_models = [provider_models]
            # Remove any models that have failed
            available_models = [
                model
                for model in provider_models
                if not any(
                    failure["provider"] == provider["name"]
                    and failure["model"] == model
                    for failure in self.failures
                )
            ]
            if available_models:
                provider_copy = provider.copy()
                provider_copy["models"] = available_models
                available_providers.append(provider_copy)
        return available_providers

    # Chat Completion wrapper
    async def chat_completions(
        self,
        prompt: ChatCompletions,
    ):
        images = []
        new_prompt = ""
        for message in prompt.messages:
            if "content" not in message:
                continue
            if isinstance(message["content"], str):
                role = message["role"] if "role" in message else "User"
                if role.lower() == "system":
                    if "/" in message["content"]:
                        new_prompt += f"{message['content']}\n\n"
                if role.lower() == "user":
                    new_prompt += f"{message['content']}\n\n"
            if isinstance(message["content"], list):
                for msg in message["content"]:
                    if "text" in msg:
                        role = message["role"] if "role" in message else "User"
                        if role.lower() == "user":
                            new_prompt += f"{msg['text']}\n\n"
                    if "image_url" in msg:
                        url = (
                            msg["image_url"]["url"]
                            if "url" in msg["image_url"]
                            else msg["image_url"]
                        )
                        if url:
                            images.append(url)
        response = await self.inference(
            new_prompt,
            images=images,
            **prompt.model_dump(),
        )
        prompt_tokens = get_tokens(str(new_prompt))
        completion_tokens = get_tokens(str(response))
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        res_model = {
            "id": prompt.user,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": prompt.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(response),
                    },
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        return res_model
