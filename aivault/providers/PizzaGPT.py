from aiohttp import ClientSession
from typing import List, Dict, Any


class PizzaGPTProvider:
    url = "https://www.pizzagpt.it"
    models = ["gpt-4o", "gpt-3.5-turbo"]

    def __init__(self, model: str = "gpt-4o", **kwargs):
        self.model = model
        self.params = kwargs

    async def inference(
        self, messages: List[Dict[str, Any]], proxy: str = None, **kwargs
    ) -> str:
        payload = {"question": messages[-1]["content"]}
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
            "Origin": self.url,
            "Referer": f"{self.url}/en",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "X-Secret": "Marinara",
        }

        async with ClientSession() as session:
            async with session.post(
                f"{self.url}/api/chatx-completion",
                json=payload,
                proxy=proxy,
                headers=headers,
            ) as response:
                response.raise_for_status()
                response_json = await response.json()
                return response_json["answer"]["content"]
