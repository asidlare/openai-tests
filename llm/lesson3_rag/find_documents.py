from pydantic import BaseModel, Field, conlist
from typing import List
from llm.utils import get_response_with_instructor
from llm.rag_utils import extract_context_table_id_and_title
import pprint


def find_tables(question, document_id):
    class Answer(BaseModel):
        relevant_table_ids: conlist(int, min_length=1, max_length=5) = Field(
            ...,
            description="Lista identyfikatorów tabel, które są najbardziej odpowiednie do odpowiedzi na pytanie (od 1 do 5 tabel)."
        )


    class TableQueryContext(BaseModel):
        question: str = Field(..., description="Pytanie, na które chcemy uzyskać odpowiedź.")
        reasoning_steps: List[str] = Field(
            ...,
            description="Lista rozważań lub kroków myślowych, które pomagają określić najlepsze tabele do odpowiedzi na pytanie."
        )
        answer: Answer = Field(
            ...,
            description="Struktura zawierajaca pola pozwalajace odpowiedzieć na zadane pytanie"
        )


    CONTEXT_ID_TITLE = extract_context_table_id_and_title(document_id)

    system_prompt = f"""
    Jesteś ekspertem w analizie sprawozdań finansowych. 
    Twoim zadaniem jest wybrać maksymalnie 5 tabel, które najlepiej odpowiadają na pytanie, korzystając z listy tabel zawierającej:
    - ID Tabeli: Unikalny identyfikator.
    - Tytuł (title): Nazwa tabeli wskazująca jej zawartość.

    Twoje zadanie:
    1. Przeanalizuj pytanie, identyfikując kluczowe elementy (np. zakres czasowy, rodzaj danych).
    2. Na podstawie tytułów wybierz najbardziej odpowiednie tabele.
    3. Uzasadnij każdy wybór.

    Tabele:
    {CONTEXT_ID_TITLE}

    """

    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    response = get_response_with_instructor(
        messages=messages,
        response_model=TableQueryContext,
        temperature=0,
        max_tokens=10000
    )

    return response


if __name__ == "__main__":
    question = "Jakie są główne źródła przychodów i jakie czynniki mają największy wpływ na ich zmienność?"
    found_tables = find_tables(question, document_id=0)
    pprint.pprint(found_tables.model_dump())
