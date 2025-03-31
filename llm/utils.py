from openai import OpenAI
from dotenv import dotenv_values


config = dotenv_values(".env")


client = OpenAI(
    api_key=config["OPENAI_API_KEY"],
)


def get_simple_response(messages, model="gpt-4o-mini-2024-07-18", temperature=0, max_tokens=200):
    completion = client.chat.completions.create(
        model=model,
        store=True,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return completion.choices[0].message.content
