from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import logging

from llm.utils import get_response_with_instructor


formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Format for the date/time
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()

# Set the console handler's level to INFO
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)


class ContractStatus(Enum):
    COLLECTING_DATA = "collecting_data"
    GENERATING = "generating"
    AUDITING = "auditing"
    REVISING = "revising"
    COMPLETED = "completed"
    ERROR = "error"


class Address(BaseModel):
    street: str = Field(..., description="Nazwa ulicy wraz z numerem")
    city: str = Field(..., description="Nazwa miejscowości")
    #state: str = Field(..., description="Województwo")
    postal_code: str = Field(..., description="Kod pocztowy")


class Party(BaseModel):
    name: str = Field(..., description="Nazwa osoby fizycznej lub prawnej")
    address: Address = Field(..., description="Adres strony")
    id_number: str = Field(..., description="PESEL/NIP/REGON")
    phone: Optional[str] = Field(None, description="Numer telefonu")


class LeaseDuration(BaseModel):
    length: int = Field(12, description="Długość wynajmu, liczba jednostek czasu")
    step: Literal['month', 'year'] = Field("month", description="Jednostka czasu, np. 'miesiąc' lub 'rok'")
    is_indefinite: bool = Field(False, description="Czy czas trwania umowy jest nieoznaczony")


class Property(BaseModel):
    address: Address = Field(..., description="Adres nieruchomości")
    condition: Optional[str] = Field(None, description="Ocena stanu nieruchomości przy przekazaniu")
    equipment: List[str] = Field(..., description="Lista wyposażenia i elementów stałych w nieruchomości")
    intended_use: Optional[Literal['residential', 'commercial', 'industrial', 'mixed-use', 'recreational']] = Field(
        "residential",
        description="Cel/sposób używania nieruchomości (np. mieszkalna, komercyjna, przemysłowa, wielofunkcyjna, rekreacyjna)"
    )


class Deposit(BaseModel):
    amount: int = Field(..., description="Wysokość kaucji w jednostkach waluty, np. 3000")
    currency: Literal["PLN", "EUR"] = Field("PLN", description="Waluta kaucji, np. PLN")
    type: Literal["jednorazowa", "wielokrotna"] = Field(
        "jednorazowa",
        description="Rodzaj kaucji: 'jednorazowa' lub 'wielokrotna'"
    )
    conditions: Optional[str] = Field(
        None,
        description="Dodatkowe warunki dotyczące kaucji, np. 'zwrot do 10 dni po zakończeniu umowy'"
    )


class Rent(BaseModel):
    amount: int = Field(..., description="Wysokość czynszu w jednostkach waluty, np. 2500")
    currency: Literal["PLN", "EUR"] = Field("PLN", description="Waluta czynszu, np. PLN")
    payment_schedule: Literal['monthly', 'quarterly', 'annually'] = Field(
        "monthly",
        description="Harmonogram płatności, np. 'monthly', 'quarterly', 'annually'"
    )
    payment_day: int = Field(
        10,
        ge=1,
        le=31,
        description="Dzień miesiąca, w którym płatność jest dokonywana (1-31), np. 10"
    )
    payment_method: Literal['bank_transfer', 'cash', 'credit_card'] = Field(
        "bank_transfer",
        description="Sposób płatności, np. 'bank_transfer', 'cash', 'credit_card'"
    )
    additional_fees: Optional[List[str]] = Field(
        None,
        description="Opłaty dodatkowe, np. 'opłata za wodę', 'opłata za prąd'"
    )
    deposit: Deposit = Field(None, description="Warunki dotyczące kaucji")


class ContractData(BaseModel):
    """Model danych umowy"""
    lessor: Party = Field(..., description="Dane wynajmującego")
    lessee: Party = Field(..., description="Dane najemcy")
    property_details: Property = Field(..., description="Szczegóły nieruchomości")
    lease_duration: LeaseDuration = Field(..., description="Okres wynajmu")
    rent_details: Rent = Field(..., description="Szczegóły dotyczące czynszu")


class Risk(BaseModel):
    chain_of_thought: List[str] = Field(
        ...,
        description="Kroki wyjaśniające prowadzące do zidentyfikowanego ryzyka. Te kroki powinny dostarczać szczegółowego uzasadnienia wykrycia problemu."
    )
    content: str = Field(
        ...,
        description="Opis zidentyfikowanego ryzyka. Powinien jasno i zwięźle podsumowywać problem."
    )
    suggested_changes: List[str] = Field(
        ...,
        description="Propozycje zmian w celu zmniejszenia lub wyeliminowania zidentyfikowanego ryzyka. Każda propozycja powinna być jasna, konkretna i możliwa do wdrożenia."
    )


class AuditResult(BaseModel):
    is_approved: bool = Field(False, description="Czy przeszło adut bez żadnych uwag?")
    risks: List[Risk] = Field(
        ...,
        description="Lista wszystkich zidentyfikowanych ryzyk. Każde ryzyko zawiera szczegółowe uzasadnienie oraz podsumowanie opisu."
    )
    #timestamp: datetime = Field(datetime.now, description="Kiedy był robiony audyt")


class ProcessMetadata(BaseModel):
    """Model metadanych procesu"""
    status: ContractStatus = Field(default=ContractStatus.COLLECTING_DATA)
    llm_history: List[Dict[str, str]] = Field(default=[])
    current_version: int = Field(default=1)
    max_revision_attempts: int = Field(default=3)
    current_revision_attempt: int = Field(default=0)
    audit_history: List[AuditResult] = Field(default_factory=list)
    process_start_time: datetime = Field(default_factory=datetime.now)
    last_update_time: datetime = Field(default_factory=datetime.now)


class ProcessContext(BaseModel):
    """Kontekst całego procesu łączący dane umowy i metadane"""
    contract_data: Optional[ContractData] = Field(None)
    contract_text: Optional[str] = Field(None)
    metadata: ProcessMetadata = Field(ProcessMetadata())


class BaseAgent(ABC):
    def __init__(self, context: ProcessContext, logger=logger):
        self.context = context
        self.logger = logger

    def _get_response(self, messages, response_model, temparature=0, max_tokens=16000):
        return get_response_with_instructor(
            messages=messages,
            response_model=response_model,
            temperature=temparature,
            max_tokens=max_tokens,
        )

    def _get_api_call(self, system_prompt:str, response_model, user_prompt: str = None):
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        return self._get_response(messages, response_model)

    @abstractmethod
    def run(self) -> bool:
        pass
