# Przygotuj składniki i kroki wykonania przepisu:
# naleśniki keto z serkiem mascarpone

from enum import Enum
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from llm.utils import get_response_with_response_format

import pprint


class Quantity(str, Enum):
    kg = 'kg'
    g = 'g'
    l = 'l'
    ml = 'ml'
    szt = 'szt'


class Ingredient(BaseModel):
    name: str = Field(..., description="Nazwa składnika potrawy")
    quantity: float = Field(..., description="Ilość składnika potrawy")
    unit: Quantity = Field(
        ...,
        description="""
        Jednostka miary zgodnie z wytycznymi (ogranicz się do: kg, g, l, ml, szt):
        - mąka, ser, produkty sypkie -> kg (kilogram)
        - przyprawy i produkty słodzące -> g (gram)
        - mleko, olej i płyny -> l (litr)
        - jajka -> szt (sztuka)
        """
    )


class Step(BaseModel):
    step_number: int = Field(..., description="Numer kroku")
    description: str = Field(..., description="Opis kroku przygotowania")
    duration: float = Field(..., description="Czas trwania kroku (w minutach)")
    additional_notes: Optional[str] = Field(
        ...,
        description="Dodatkowe uwagi dotyczące kroku (np. 'Wymieszaj jajka z serkiem mascarpone na gładka masę')"
    )


class Recipe(BaseModel):
    name: str = Field(..., description="Nazwa potrawy")
    ingredients: List[Ingredient] = Field(description="Lista składników")
    steps: List[Step] = Field(description="Lista kroków do wykonania")


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

    return get_response_with_response_format(
        messages=messages,
        response_format=Recipe,
        max_tokens=1000
    )


if __name__ == "__main__":
    response = create_recipe("przepis na naleśniki keto z serkiem mascarpone")
    pprint.pprint(json.loads(response))
