# Maybe address - if not a valid address explains why

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from llm.utils import get_response_with_instructor

import re
import pprint


class PostalCode(BaseModel):
    value: str

    @field_validator('value')
    def must_contains_hyphen(cls, v):
        if "-" not in v:
            raise ValueError(f"`-` was found in the post code `{v}`")
        return v

    @field_validator('value')
    def must_be_valid_postal_code(cls, v):
        if not re.match(r'^\d{2}-\d{3}$', v):
            raise ValueError(f"Invalid postal code format: `{v}`. Expected format is `XX-XXX`.")
        return v


class Address(BaseModel):
    street: str = Field(..., description="Street name")
    house_number: str = Field(..., description="House number")
    house_unit_number: Optional[str] = Field(..., description="Flat number if it is a part of the address")
    city: str = Field(..., description="City or village name")
    state: str = Field(..., description="State or province name")
    postal_code: PostalCode = Field(..., description="Postal code")


class MaybeAddress(BaseModel):
    """
    It represents the result of operation, which can return an address
    together with metadata about success or failure.
    """
    result: Optional[Address] = Field(
        default=None,
        description="Information about address if it is available"
    )
    error: bool = Field(
        default=False,
        description="Indicates if an error occurred during the operation"
    )
    message: Optional[str] = Field(
        default=None,
        description="Message including information about failure reasons if an error occurred"
    )


def recognize_address(address):
    messages = [
        {"role": "system", "content": "Recognize address. If it is not a valid address, explain why."},
        {"role": "user", "content": address}
    ]
    return get_response_with_instructor(
        messages=messages,
        response_model=MaybeAddress,
    )


if __name__ == "__main__":
    address = "Warszawa, Wiśniowa 10/15, 11-100, mazowieckie"
    response = recognize_address(address)
    pprint.pprint(response.model_dump())
    print('*' * 50)

    address = "smok wawelski"
    response = recognize_address(address)
    pprint.pprint(response.model_dump())
    print('*' * 50)

    address = "Podaj adres smoka wawelskiego."
    response = recognize_address(address)
    pprint.pprint(response.model_dump())
    print('*' * 50)
