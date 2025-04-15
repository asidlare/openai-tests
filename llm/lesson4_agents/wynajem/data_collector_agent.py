from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

from llm.lesson4_agents.wynajem.base_agent import (
    BaseAgent,
    Party,
    Property,
    LeaseDuration,
    Rent,
)



class ThoughtStep(BaseModel):
    """
    Represents a step in the thought-action process during collecting data.

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


class ContractStatus(Enum):
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
    Represents missing information required to complete a contract data.

    Attributes:
        field (str): The specific field (e.g., address) that is missing.
        question (str): A question prompting the user for the missing information.
    """
    field: str = Field(
        ...,
        description="Nazwa brakującego pola w zamówieniu, np. 'address'."
    )
    question: str = Field(
        ...,
        description="Pytanie, które należy zadać użytkownikowi, aby uzupełnić brakujące informacje."
    )


class ContractData(BaseModel):
    """Model danych umowy"""
    lessor: Optional[Party] = Field(..., description="Dane wynajmującego: imię, nazwisko, ulica z numerem, kod pocztowy, miasto, id")
    lessee: Optional[Party] = Field(..., description="Dane najemcy: imię, nazwisko, ulica z numerem, kod pocztowy, miasto ,id")
    property_details: Optional[Property] = Field(..., description="Szczegóły nieruchomości: ulica z numerem, kod pocztowy, miasto")
    lease_duration: Optional[LeaseDuration] = Field(..., description="Okres wynajmu")
    rent_details: Optional[Rent] = Field(..., description="Szczegóły dotyczące czynszu")


class ContractAnalysis(BaseModel):
    """
    Represents the analysis and breakdown of a current pizza order.

    Attributes:
        thoughts (list[ThoughtStep]): Reasoning and decisions made during the process.
        current_contract_data (ContractData): The current pizza order being processed.
        status (str): The status of the current order (e.g., incomplete, complete).
        missing_info (list[MissingInfo]): Information required to complete the order.
        confirmation_message (str): Message confirming the order details.
        resignation_message (str): Message indicating order abandonment.
    """
    thoughts: List[ThoughtStep] = Field(
        ...,
        description="Lista kroków myślowych i działań wykonanych przez asystenta. Rozważaj do czasu uzupełnienia danych."
    )
    current_contract_data: Optional[ContractData] = Field(
        ...,
        description="Obecne dane do wynajmu nieruchomości."
    )
    status: ContractStatus = Field(
        ...,
        description="Aktualny status danych: INCOMPLETE, COMPLETE lub CONFIRMED."
    )
    missing_info: Optional[List[MissingInfo]] = Field(
        None,
        description="Lista brakujących danych do umowy najmu. Pytaj do czasu aż zostaną uzupełnione dane do umowy najmu."
    )
    confirmation_message: Optional[str] = Field(
        None,
        description="Komunikat potwierdzający dame, jeśli zostało potwierdzone (CONFIRMED)."
    )


contract_setup_knowledge_map = """
ELEMENTY OBOWIĄZKOWE:
1. Dane stron umowy (wynajmujący i najemca)
2. Przedmiot najmu (dokładne określenie nieruchomości)
3. Czas trwania umowy:
   - oznaczony lub nieoznaczony
   - przy czasie dłuższym niż rok - obowiązkowo forma pisemna
4. Wysokość czynszu:
   - kwota
   - termin i sposób płatności
"""


class DataCollectorAgent(BaseAgent):
    def __init__(self, context):
        super().__init__(context)
        self.conversation_history = []
        system_prompt = f"""
            Jesteś asystentem wspierajacym proces zbierania danych dla umowy najmu nieruchomości

            {contract_setup_knowledge_map}
        """
        self.conversation_history.append({"role": "system", "content": system_prompt})
        self.current_contract_data = None

    def run(self) -> bool:
        self.logger.info("[DataCollectorAgent] Collecting data...")

        print("Witaj! Podaj dane do umowy najmu:")

        while True:
            user_input = input("\nTy: ")

            if user_input.lower() in ['koniec', 'wyjdź', 'exit']:
                print("Proces podawania danych został zakończony!")
                return True

            response = self.process_input(user_input)
            print(f"\nAsystent: {response}")

            if "complete" in response:
                return True

    def process_input(self, user_input: str) -> str:

        self.conversation_history.append({"role": "user", "content": user_input})
        analysis = self.process_data_collection(user_input)

        for t in analysis.thoughts:
            self.logger.info(t.thought)
            self.logger.info(f"{t.action}: {t.action_input}")
        self.logger.info(f"current_contract_data={analysis.current_contract_data}")
        self.logger.info(f"status={analysis.status}")

        self.current_contract_data = analysis.current_contract_data

        if analysis.status == ContractStatus.INCOMPLETE:
            response = analysis.missing_info[0].question
        elif analysis.status == ContractStatus.COMPLETE:
            self.context.contract_data = self.current_contract_data
            response = analysis.status.value

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def process_data_collection(self, user_input: str) -> ContractAnalysis:

        current_contract_data = ""
        if self.current_contract_data:
            current_contract_data = f"\nAktualne dane do umowy:\n{self.current_contract_data.model_dump()}"
        self.conversation_history.append({
            "role": "user",
            "content": f"{user_input} {current_contract_data}"
        })

        return self._get_response(messages=self.conversation_history, response_model=ContractAnalysis)
