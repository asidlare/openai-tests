from llm.utils import get_simple_response


def simple_query():
    messages = [
        {"role": "user", "content": "What is your name?"},
    ]

    return get_simple_response(messages)


if __name__ == "__main__":
    result = simple_query()
    print('*' * 50, result)
