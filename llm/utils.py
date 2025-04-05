from openai import OpenAI
from dotenv import dotenv_values


config = dotenv_values(".env")


client = OpenAI(
    api_key=config["OPENAI_API_KEY"],
)


def get_simple_response(
    messages,
    model="gpt-4o-mini-2024-07-18",
    temperature=0,
    max_tokens=200
):
    response = client.chat.completions.create(
        model=model,
        store=True,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def get_response_with_tool_call(
    messages,
    tools,
    model="gpt-4o-mini-2024-07-18",
    tool_choice="auto",
    temperature=0,
    max_tokens=200,
):
    response = client.chat.completions.create(
        model=model,
        store=True,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.tool_calls
