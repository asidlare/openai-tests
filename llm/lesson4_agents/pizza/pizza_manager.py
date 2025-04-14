from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from loguru import logger
import sys

from llm.utils import get_response_with_instructor

logger.remove()

logger.level("thought", no=35, color="<red>")
logger.level("action", no=25, color="<blue>")


# Add methods to logger
def thought(message, *args, **kwargs):
    logger.log("thought", message, *args, **kwargs)


def action(message, *args, **kwargs):
    logger.log("action", message, *args, **kwargs)


logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>"
)

# Attach custom methods to the logger
logger.thought = thought
logger.action = action


"""
Pizza Manager Module

This module handles the process of managing pizza orders, including user inputs,
order validations, processing pizza options, and generating order summaries.

**Core Functionalities:**
- Supports selection of pizza size, crust type, and toppings.
- Validates and processes orders to generate a structured summary.
- Maintains conversation history for a seamless user experience.
- Handles missing information and status updates efficiently.

**Main Components:**
1. Data Classes:
   - `PizzaTopping`: Represents a topping and its associated attributes.
   - `PizzaOrder`: Encapsulates a pizza order with size, crust, toppings, and price.
   - `OrderAnalysis`: Tracks the current order, missing info, and order status.
2. Functions:
   - `process_pizza_order`: Handles the step-by-step creation or update of an order.
   - `run_react`: Simulates a reaction-based interface for pizza order processing.
3. Manager:
   - `PizzaOrderManager`: Core class for handling user interactions and generating order summaries.

Dependencies:
- Standard Python libraries for string and data manipulations.
"""


"""
A mapping or knowledge base for resolving user inputs to system actions.
"""
MAP_KNOWLEDGE = """
Zamówienie wymaga następujących informacji:
- Rozmiar pizzy
- Rodzaj ciasta
- Składniki

Przeanalizuj zamówienie i:
1. Jeśli brakuje informacji, określ status jako INCOMPLETE i przygotuj pytania o brakujące dane.
2. Zadawaj pytania użytkownikowi na podstawie przygotowanych przez ciebie pytań do momentu,
   kiedy uzyskasz potrzebne informacje.
3. Jeśli wszystkie informacje są kompletne, określ status jako COMPLETE i przygotuj podsumowanie.
4. Jeśli zamówienie jest potwierdzone, oznacz jako CONFIRMED.
5. Jeśli zamówienie nie zostanie potwierdzone, podziękuj za wizytę i zaproś na następną.

Cennik:
- Mała pizza: 25 zł
- Średnia pizza: 30 zł
- Duża pizza: 35 zł
- Podwójny składnik: +8 zł
- Pojedyńczy składnik: +5 zł
- Grube ciasto: +3 zł
"""


class ThoughtStep(BaseModel):
    """
    Represents a step in the thought-action process during order management.

    Attributes:
        thought (str): The thought process or reasoning for the step.
        action (str): The corresponding action taken for the thought process.
        action_input (str): Input data used for processing the action.
    """
    thought: str = Field(
        ...,
        description="Tok myślenia asystenta, który prowadzi do wykonania zleconego zadania."
    )
    action: str = Field(
        ...,
        description="Nazwa zadania, którym asystent się zajmuje, np. 'ask' lub 'calculate'."
    )
    action_input: str = Field(
        ...,
        description="Dane wejściowe dla danego zadania, np. treść pytania lub dane do obliczeń."
    )


class OrderStatus(Enum):
    """
    Enumerates the possible statuses of an order.

    Attributes:
        INCOMPLETE (str): Denotes that the order is incomplete.
        COMPLETE (str): Denotes that the order is successfully completed.
        CONFIRMED (str): Denotes that the order has been confirmed by the user.
    """
    INCOMPLETE = "incomplete"  # niekompletne
    COMPLETE = "complete"      # kompletne i gotowe do potwierdzenia
    CONFIRMED = "confirmed"    # potwierdzone


class MissingInfo(BaseModel):
    """
    Represents missing information required to complete a pizza order.

    Attributes:
        field (str): The specific field (e.g., size, crust) that is missing.
        question (str): A question prompting the user for the missing information.
    """
    field: str = Field(
        ...,
        description="Nazwa brakującego pola w zamówieniu, np. 'size' lub 'crust_type'."
    )
    question: str = Field(
        ...,
        description="Pytanie, które należy zadać użytkownikowi, aby uzupełnić brakujące informacje."
    )


class PizzaTopping(BaseModel):
    """
    Represents a pizza topping and its associated properties.

    Attributes:
        name (str): The name of the topping (e.g., mozzarella, pepperoni).
        extra (bool): Indicates if the topping is an additional or premium option.
    """
    name: str = Field(
        ...,
        description="Nazwa składnika pizzy, np. 'ser', 'pepperoni'."
    )
    extra: bool = Field(
        False,
        description="Flaga oznaczająca, czy składnik ma być podwójny (True) czy pojedynczy (False)."
    )


class PizzaOrder(BaseModel):
    """
    Represents a complete pizza order.

    Attributes:
        size (str): The size of the pizza (e.g., small, medium, large).
        toppings (list[PizzaTopping]): List of selected pizza toppings.
        crust_type (str): The type of pizza crust (e.g., thin, deep dish).
        total_price (float): The total price of the order.

    Methods:
        __init__(...): Initializes the order with default or provided details.
    """
    size: Optional[str] = Field(
        None,
        description="Rozmiar pizzy, jeden z dostępnych: 'mała', 'średnia', 'duża'."
    )
    toppings: List[PizzaTopping] = Field(
        default_factory=list,
        description="Lista wybranych składników pizzy."
    )
    crust_type: Optional[str] = Field(
        None,
        description="Rodzaj ciasta, jeden z dostępnych: 'cienkie', 'grube'."
    )
    total_price: Optional[float] = Field(
        None,
        description="Całkowita cena zamówienia w złotówkach."
    )


class OrderAnalysis(BaseModel):
    """
    Represents the analysis and breakdown of a current pizza order.

    Attributes:
        thoughts (list[ThoughtStep]): Reasoning and decisions made during the process.
        current_order (PizzaOrder): The current pizza order being processed.
        status (str): The status of the current order (e.g., incomplete, complete).
        missing_info (list[MissingInfo]): Information required to complete the order.
        confirmation_message (str): Message confirming the order details.
        resignation_message (str): Message indicating order abandonment.
    """
    thoughts: List[ThoughtStep] = Field(
        ...,
        description="Lista kroków myślowych i działań wykonanych przez asystenta."
    )
    current_order: PizzaOrder = Field(
        ...,
        description="Obecne zamówienie pizzy z wybranymi opcjami."
    )
    status: OrderStatus = Field(
        ...,
        description="Aktualny status zamówienia: INCOMPLETE, COMPLETE lub CONFIRMED."
    )
    missing_info: Optional[List[MissingInfo]] = Field(
        None,
        description="Lista brakujących informacji w zamówieniu."
    )
    confirmation_message: Optional[str] = Field(
        None,
        description="Komunikat potwierdzający zamówienie, jeśli zostało potwierdzone (CONFIRMED)."
    )
    resignation_message: Optional[str] = Field(
        None,
        description="Komunikat wyświetlany kiedy zamówienie nie zostanie potwierdzone."
    )


PIZZA_SIZES = ["mała", "średnia", "duża"]
CRUST_TYPES = ["cienkie", "grube"]
AVAILABLE_TOPPINGS = [
    "ser",
    "szynka",
    "papryka",
    "oliwki",
    "cebula",
    "salami",
    "pieczarki",
    "rucola",
    "pepperoni"
]


def process_pizza_order(user_input: str, current_order: Optional[PizzaOrder] = None) -> OrderAnalysis:
    """
    Processes the current pizza order and verifies all required fields.

    Parameters:
        pizza_order (PizzaOrder): The current order to be finalized.

    Returns:
        str: Confirmation details or error messages based on the order validation.
    """
    system_prompt = f"""
    Jesteś asystentem w pizzerii, który specjalizuje się w odbieraniu zamówień od klientów.

    {MAP_KNOWLEDGE}
    """

    current_order_info = ""
    if current_order:
        current_order_info = f"\nAktualne zamówienie:\n{current_order.model_dump_json()}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_input}{current_order_info}"}
    ]

    response = get_response_with_instructor(
        messages=messages,
        response_model=OrderAnalysis,
        temperature=0,
        max_tokens=16000
    )

    return response


class PizzaOrderManager:
    """
    Manages the creation and processing of pizza orders through user interaction.

    Attributes:
        current_order (PizzaOrder): The active pizza order being managed.
        conversation_history (list[str]): Tracks user inputs and system responses.

    Methods:
        __init__(): Initializes the manager with an empty order and conversation history.
        process_input(user_input: str) -> str:
            Handles user inputs to update the current order or retrieve missing info.
        _generate_summary() -> str:
            Generates a detailed summary of the current order, including price breakdown.
    """
    def __init__(self):
        self.current_order = PizzaOrder()
        self.conversation_history = []

    def process_input(self, user_input: str) -> str:

        self.conversation_history.append(("user", user_input))
        analysis = process_pizza_order(user_input, self.current_order)

        for t in analysis.thoughts:
            logger.thought(t.thought)
            logger.action(f"{t.action}: {t.action_input}")
        logger.info(f"current_order={analysis.current_order}")
        logger.info(f"status={analysis.status}")

        self.current_order = analysis.current_order

        if analysis.status == OrderStatus.INCOMPLETE:
            response = analysis.missing_info[0].question
        elif analysis.status == OrderStatus.COMPLETE:
            response = self._generate_summary() + "\nCzy potwierdzasz zamówienie? (tak/nie)"
        elif user_input.lower() == 'nie':
            response = analysis.resignation_message
        else: # CONFIRMED
            response = analysis.confirmation_message

        self.conversation_history.append(("assistant", response))
        return response

    def _generate_summary(self) -> str:
        order = self.current_order
        summary = f"\nPodsumowanie zamówienia:\n"
        summary += f"Rozmiar: {order.size}\n"
        summary += f"Rodzaj ciasta: {order.crust_type}\n"
        summary += "Składniki:\n"
        for topping in order.toppings:
            extra = "(podwójnie)" if topping.extra else ""
            summary += f"- {topping.name} {extra}\n"
        summary += f"Całkowita cena: {order.total_price} zł"
        return summary


def run_react() -> str:
    """
    Simulates a reaction-based interface for interactive pizza order processing.

    Returns:
        str: The final pizza order confirmation message or action result.
    """
    order_manager = PizzaOrderManager()
    print("Witaj w pizzerii! Złóż swoje zamówienie:")

    while True:
        user_input = input("\nTy: ")

        if user_input.lower() in ['koniec', 'wyjdź', 'exit']:
            print("Dziękujemy za wizytę!")
            break

        response = order_manager.process_input(user_input)
        print(f"\nAsystent: {response}")

        if "Zamówienie zostało potwierdzone" in response:
            break


if __name__ == '__main__':
    run_react()
