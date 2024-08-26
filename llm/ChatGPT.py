from openai import OpenAI


class ChatGPT():
    def __init__(self, model_path='gpt-3.5-turbo', api_key=None, openai_base_url=None):
        self.openai_client = OpenAI(api_key=api_key, base_url=openai_base_url)
        self.model_path = model_path

    def chat(self, message):
        chat_completion = self.openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ],
            model=self.model_path,
        )

        return chat_completion.to_dict()['choices'][0]['message']['content']
