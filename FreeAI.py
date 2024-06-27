import logging
import random
import importlib
import os


def get_providers():
    providers = []
    for file in os.listdir("providers"):
        if file.endswith(".py") and not file.startswith("__"):
            module = importlib.import_module(f"providers.{file[:-3]}")
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


class FreeAI:
    def __init__(self, **kwargs):
        self.providers = get_providers()
        self.provider = random.choice(self.providers)["class"]()
        self.provider_name = self.provider.__class__.__name__
        self.friendly_provider_name = str(self.provider_name).split("Provider")[0]
        self.model = random.choice(self.provider.models)
        self.failures = []
        self.provider_failure_count = 0

    async def inference(self, prompt, images: list = []):
        logging.info(
            f"[FreeAI] Using provider: {self.friendly_provider_name} with model: {self.model}"
        )
        try:
            return await self.provider.inference(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logging.error(f"[FreeAI] {e}")
            self.failures.append({"provider": self.provider_name, "model": self.model})
            if len(self.failures) < len(self.providers):
                available_providers = self.get_available_providers()
                if available_providers:
                    provider = random.choice(available_providers)
                    self.provider = provider["class"]()
                    self.provider_name = provider["name"]
                    self.model = random.choice(provider["models"])
                    logging.info(
                        f"[FreeAI] Switching to provider: {self.friendly_provider_name} with model: {self.model}"
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
