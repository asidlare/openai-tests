# naleśniki keto z serkiem mascarpone

from llm.utils import get_response_with_instructor
from llm.lesson2.pancakes_with_response_format import Recipe
import pprint


def create_recipe(name: str) -> Recipe:
    messages = [
        {
            "role": "system",
            "content": """
                Jesteś asystentem potrafiącym doradzać w sprawach kulinarnych.
                Podajesz przepisy kulinarne, zgodnie z wytycznymi z zapytania.
                Jeżeli pytanie nie dotyczy przepisu kulinarnego, odpowiedz:
                'Potrafię odpowiadać tylko na pytania o przepisy kulinarne'
            """
        },
        {
            "role": "user",
            "content": name
        },
    ]

    return get_response_with_instructor(
        messages=messages,
        response_model=Recipe,
        max_tokens=1000
    )


if __name__ == "__main__":
    response = create_recipe("przepis na naleśniki keto z serkiem mascarpone")
    pprint.pprint(response.model_dump())