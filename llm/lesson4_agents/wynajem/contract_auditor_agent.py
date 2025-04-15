from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional
from llm.lesson4_agents.wynajem.base_agent import (
    BaseAgent,
    AuditResult,
    Risk,
)
from llm.utils import get_audit_checklist


class Risk(BaseModel):
    chain_of_thought: List[str] = Field(
        ...,
        description="""
            Kroki wyjaśniające prowadzące do zidentyfikowanego ryzyka.
            Te kroki powinny dostarczać szczegółowego uzasadnienia wykrytego problemu.
        """
    )
    content: str = Field(
        ...,
        description="Opis zidentyfikowanego ryzyka. Powinien jasno i zwięźle podsumowywać problem niezgodny z polskim prawem"
    )


class AuditResult(BaseModel):
    is_approved: bool = Field(False, description="Czy przeszło audyt bez żadnych uwag?")
    risks: List[Risk] = Field(
        ...,
        description="Lista wszystkich zidentyfikowanych ryzyk. Każde ryzyko zawiera szczegółowe uzasadnienie oraz podsumowanie opisu."
    )


class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class Category(Enum):
    ESSENTIAL_ELEMENTS = "Elementy konieczne umowy"
    FORM = "Forma umowy"
    DURATION = "Czas trwania"
    OBLIGATIONS = "Obowiązki stron"
    PAYMENTS = "Płatności"
    TERMINATION = "Wypowiedzenie i zakończenie"
    MAINTENANCE = "Utrzymanie i naprawy"
    RIGHTS = "Prawa stron"
    OTHER = "Inne"


class LegalReference(BaseModel):
    """
    Reprezentuje odniesienie do przepisu prawnego.
    """
    article: str = Field(..., description="Numer artykułu (np. 'Art. 659')")
    paragraph: Optional[str] = Field(None, description="Numer paragrafu")
    description: str = Field(..., description="Krótki opis treści przepisu")


class Check(BaseModel):
    """
    Reprezentuje pojedynczy punkt kontrolny w audycie.
    """
    question: str = Field(..., description="Pytanie kontrolne związane z audytem.")
    legal_basis: List[LegalReference] = Field(..., description="Podstawa prawna")
    possible_issues: List[str] = Field(
        ...,
        description="Lista potencjalnych problemów, które mogą wynikać z braku spełnienia wymogu."
    )
    priority: Optional[Priority] = Field(None,
                                         description="Priorytet pytania (niższa wartość oznacza wyższy priorytet).")
    category: Optional[Category] = Field(None, description="Kategoria pytania, np. 'Płatności'.")
    note: Optional[str] = Field(None, description="Dodatkowy komentarz")
    validation_hint: Optional[str] = Field(None, description="Wskazówka jak zweryfikować zgodność")


class AuditChecklist(BaseModel):
    """
     Reprezentuje listę kontrolną dla audytu umowy.
    """
    questions: List[Check] = Field(..., description="Lista pytań kontrolnych do przeprowadzenia audytu.")
    checklist_name: str = Field(..., description="Nazwa checklisty.")
    version: Optional[str] = Field(1.0, description="Opcjonalna wersja checklisty (np. '1.0').")
    applicable_law_version: str = Field(..., description="Wersja przepisów prawnych")


def audit_checklist():
    audit_checklist_dict = get_audit_checklist()
    return AuditChecklist(**audit_checklist_dict)


class ContractAuditorAgent(BaseAgent):
    def run(self) -> bool:
        self.logger.info(f"[ContractAuditorAgent] Auditing contract... Version: {self.context.metadata.current_version}")

        audit_result = self._perform_audit()
        self.context.metadata.audit_history.append(audit_result)
        self.context.metadata.last_update_time = datetime.now()

        if not audit_result.is_approved:
            self.verbose(audit_result)

        return audit_result.is_approved

    def verbose(self, result):
        for r in result.risks:
            for c in r.chain_of_thought:
                self.logger.info(c)
            self.logger.info(f"Problem: {r.content}\n")

    def _run_audit(self):
        system_prompt = """
            Jesteś ekspertem prawnym AI specjalizującym się w analizie umów. 
            Twoim zadaniem jest analiza tekstu umowy dostarczonego przez użytkownika, identyfikacja potencjalnych ryzyk prawnych, 
            wskazanie klauzul, które mogą być problematyczne, oraz wyjaśnienie swojego rozumowania krok po kroku. "

            Skup się na zgodności z polskim prawem, wykonalności oraz potencjalnych niejasnościach w tekście. 
            Gdzie to możliwe, dostarczaj praktyczne sugestie.
        """

        user_prompt = f"""
            Przeanalizuj poniższy tekst umowy i zidentyfikuj potencjalne ryzyka. 
            Skup się na wykrywaniu niezgodności z obowiązującym w Polsce prawem, niejasności oraz klauzul wymagających doprecyzowania. 

            CHECKLIST
            {audit_checklist().model_dump_json()}

            Tekst umowy: 
            {self.context.contract_text}
        """

        try:
            result = self._get_api_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=AuditResult
            )
            return result
        except Exception as e:
            self.logger.error(f"Error! {e}")

    def _perform_audit(self) -> AuditResult:
        if self.context.metadata.current_revision_attempt >= self.context.metadata.max_revision_attempts:
            return AuditResult(
                is_approved=False,
                risks=[
                    Risk(
                        chain_of_thought=[],
                        content="Maksymalna liczba prób przekroczona",
                        suggested_changes=[]
                    )
                ]
            )
        return self._run_audit()
        '''
        return AuditResult(
            is_approved=False,
            risks=[
                Risk(
                    chain_of_thought=[],
                    content="Potrzebne poprawki w sekcji X",
                    suggested_changes=["section_x zmienić na section_y"]
                )
            ]
        )
        '''
