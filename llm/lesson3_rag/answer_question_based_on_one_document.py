from pydantic import BaseModel, Field
from typing import List, Optional
from llm.utils import get_response_with_instructor
from llm.rag_utils import extract_tables_from_parquet_file
from llm.lesson3_rag.find_documents import find_tables
import pprint


def answer_question(question, document_id, table_ids):
    class TableSource(BaseModel):
        document_id: int = Field(..., description="ID dokumentu, który zawiera tabelę, z której pochodzi informacja.")
        table_id: int = Field(..., description="ID tabeli, z której pochodzi informacja, odnosząca się do dokumentu.")

    class AnswerFA(BaseModel):
        question: str = Field(..., description="Treść pytania, na które udzielana jest odpowiedź.")
        reasoning_steps: List[str] = Field(
            ...,
            description="Lista rozważań lub kroków myślowych, które doprowadziły do odpowiedzi."
        )
        answer: str = Field(
            ...,
            description="Odpowiedź na zadane pytanie. Odpowiedź powinna opierać się na zweryfikowanych faktach, być precyzyjna i zawierać wszystkie istotne informacje."
        )
        confidence: Optional[float] = Field(
            None,
            ge=0.0,
            le=1.0,
            description="Poziom pewności w udzielonej odpowiedzi (0.0 - niska pewność, 1.0 - wysoka pewność)."
        )
        references: List[TableSource] = Field(
            None,
            description="Lista źródeł wykorzystanych do udzielenia odpowiedzi, wskazanych przez pary ID dokumentu oraz tabeli."
        )


    CONTEXT_SELECTED_TABLES = extract_tables_from_parquet_file(table_ids, document_id)

    system_prompt = f"""
    Jesteś asystentem finansowym i ekspertem w analizie sprawozdań finansowych.

    Odpowiadaj na pytania na podstawie danych zawartych w poniższych tabelach. 
    Jeśli nie znasz odpowiedzi, powiedz "Nie wiem", nie twórz odpowiedzi na siłę.

    Tabele:
    {CONTEXT_SELECTED_TABLES}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    response = get_response_with_instructor(
        messages=messages,
        response_model=AnswerFA,
        temperature=0,
        max_tokens=10000,
    )

    return response


if __name__ == "__main__":
    question = "Jakie są główne źródła przychodów i jakie czynniki mają największy wpływ na ich zmienność?"
    found_tables = find_tables(question, document_id=0)
    pprint.pprint(found_tables.model_dump())
    print('*' * 50)

    response = answer_question(
        question=question,
        document_id=0,
        table_ids=found_tables.answer.relevant_table_ids,
    )
    pprint.pprint(response.model_dump())
