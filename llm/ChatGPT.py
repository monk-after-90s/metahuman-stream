from openai import OpenAI, AsyncOpenAI


class ChatGPT():
    def __init__(self, model_path='gpt-3.5-turbo', api_key=None, openai_base_url=None):
        self.api_key = api_key
        self.base_url = openai_base_url
        self.openai_client: None | OpenAI | AsyncOpenAI = None
        self.model = model_path

    def chat(self, message):  # todo 接收可选chatid参数以维护聊天历史
        self.openai_client = self.openai_client or OpenAI(api_key=self.api_key, base_url=self.base_url)
        chat_completion = self.openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ],
            model=self.model
        )

        return chat_completion.to_dict()['choices'][0]['message']['content']

    async def chat_stream(self, message: str):
        self.openai_client = self.openai_client or AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        stream = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            stream=True,
        )

        async for chunk in stream:
            yield (chunk.choices[0].delta.content or "")
