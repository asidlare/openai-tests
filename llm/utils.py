from openai import OpenAI
from dotenv import dotenv_values
import instructor


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


def get_response_with_response_format(
    messages,
    response_format,
    model="gpt-4o-mini-2024-07-18",
    temperature=0,
    max_tokens=200,
):
    response = client.beta.chat.completions.parse(
        model=model,
        store=True,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def get_response_with_instructor(
    messages,
    response_model,
    model="gpt-4o-mini-2024-07-18",
    temperature=0,
    max_tokens=200,
):
    instr_client = instructor.patch(client, mode=instructor.Mode.MD_JSON)

    response = instr_client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response
