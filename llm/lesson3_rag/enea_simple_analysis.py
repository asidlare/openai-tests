from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional, Set
from datetime import date
from tqdm import tqdm
import pprint
from llm.utils import get_response_with_instructor
from llm.rag_utils import documents, extract_tables


class TimePeriod(BaseModel):
    start_date: Optional[date] = Field(
        ...,
        description="Data początkowa"
    )
    end_date: Optional[date] = Field(
        ...,
        description="Data końcowa"
    )
    granularity: Optional[Literal['yearly', 'quarterly', 'monthly', 'daily']] = Field(
        ...,
        description="Granulacja czasowa"
    )


class ColumnDescription(BaseModel):
    name: str  = Field(
        ...,
        description="Nazwa kolumny"
    )
    data_type: str  = Field(
        ...,
        description='Typ danych (np. "string", "integer", "float", "date")'
    )
    unit: Optional[str] = Field(
        ...,
        description='Jednostka miary (np. "PLN", "EUR", "kg", itp.)'
    )


class TableDetails(BaseModel):
    title: str = Field(
        ...,
        description="Tytuł tabeli"
    )
    description: str = Field(
        ...,
        description="Krótki opis treści i celu tabeli"
    )
    category: str = Field(
        ...,
        description="Główna kategoria tabeli"
    )
    tags: Set[str] = Field(
        default_factory=set,
        description="Tagi opisujące zawartość"
    )
    columns: List[ColumnDescription] = Field(
        default_factory=list,
        description="Opis kolumn"
    )
    rows: int = Field(
        ...,
        description="Liczba wierszy danych w tabeli"
    )
    time_period: Optional[TimePeriod] = Field(
        ...,
        description="Zakres czasowy danych (jeśli dotyczy)"
    )
    currency: Optional[Literal['PLN', 'EUR', 'USD']] = Field(
        None,
        description="Waluta używana w tabeli (jeśli dotyczy)"
    )
    questions: List[str] = Field(
        default_factory=list,
        description="Lista 5 najważniejszych pytań, na które tabela odpowiada"
    )

    # Dodatkowe pola:
    practical_applications: List[str] = Field(
        ...,
        description="Praktyczne zastosowania danych tabeli (np. analiza rentowności, prognozowanie, optymalizacja kosztów)"
    )
    key_insights: Optional[List[str]] = Field(
        ...,
        description="Kluczowe wnioski wynikające z analizy tabeli, pomocne w podejmowaniu decyzji"
    )


def describe_table(table, response_model):
    system_prompt = """
Jesteś ekspertem finansowym specjalizującym się w analizie sprawozdań finansowych. Twoim zadaniem jest przeanalizować
dane w formie tabel i dostarczenie wymaganych informacji.

Twoje zadania obejmują:
1. Analizę danych, identyfikację kluczowych informacji i wniosków finansowych oraz biznesowych.
2. Interpretację kolumn w kontekście typów danych, jednostek miary i ich roli w analizie finansowej.
3. Formułowanie praktycznych zastosowań danych, takich jak analiza rentowności, prognozowanie, optymalizacja kosztów czy ocena ryzyka.
4. Zadawanie pytań pogłębiających analizę i wskazywanie ewentualnych luk lub problemów w danych.
5. Wskazywanie potencjalnych ograniczeń danych, takich jak brak spójności czy niekompletność.

Twoje opisy powinny być jasne, precyzyjne i skupione na praktycznych wnioskach wspierających decyzje finansowe.
"""
    response = get_response_with_instructor(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Opisz tabele jak najlepiej: \n\n {table}"},
        ],
        response_model=response_model,
        max_tokens=1000,
    )

    return response


def analyze_tables(start_idx, end_idx):

    class Table(BaseModel):
        markdown: str
        details: TableDetails


    tables = extract_tables(documents["ENEA"])

    llm_tables = []
    for raw_table in tqdm(tables[start_idx:end_idx]):
        details = describe_table(raw_table, TableDetails)
        table = Table(markdown=raw_table, details=details)
        llm_tables.append(table)

    for data in llm_tables:
        print('*' * 100)
        pprint.pprint(data.model_dump())
    print('*' * 100)


if __name__ == "__main__":
    analyze_tables(10, 13)
