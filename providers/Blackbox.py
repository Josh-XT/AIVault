import uuid
import secrets
import requests
from aiohttp import ClientSession
from typing import List, Dict, Any


class BlackboxProvider:
    url = "https://www.blackbox.ai"
    models = ["gpt-4o", "gpt-3.5-turbo"]

    def __init__(self, model: str = "gpt-4o", **kwargs):
        self.model = model
        self.params = kwargs

    async def inference(
        self,
        messages: List[Dict[str, Any]],
        proxy: str = None,
        images: List[str] = [],
        **kwargs,
    ) -> str:
        if images != []:
            messages[-1]["data"] = {
                "fileText": f"{uuid.uuid4()}.jpg",
                "imageBase64": requests.get(images[0]).content.decode("utf-8"),
            }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": self.url,
            "Content-Type": "application/json",
            "Origin": self.url,
            "DNT": "1",
            "Sec-GPC": "1",
            "Alt-Used": "www.blackbox.ai",
            "Connection": "keep-alive",
        }

        async with ClientSession(headers=headers) as session:
            random_id = secrets.token_hex(16)
            random_user_id = str(uuid.uuid4())
            data = {
                "messages": messages,
                "id": random_id,
                "userId": random_user_id,
                "codeModelMode": True,
                "agentMode": {},
                "trendingAgentMode": {},
                "isMicMode": False,
                "isChromeExt": False,
                "playgroundMode": False,
                "webSearchMode": False,
                "userSystemPrompt": "",
                "githubToken": None,
            }

            async with session.post(
                f"{self.url}/api/chat", json=data, proxy=proxy
            ) as response:
                response.raise_for_status()
                content = ""
                async for chunk in response.content.iter_chunked(1024):
                    if chunk:
                        content += chunk.decode()
                if "$@$" in content:
                    # Count how many times $@$ appears in the content
                    count = content.count("$@$")
                    if count == 1:
                        content = content.split("$@$")[1]
                    elif count == 2:
                        content = content.split("$@$")[2]
                return content
