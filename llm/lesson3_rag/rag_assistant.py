from pydantic import BaseModel, Field, conlist, PrivateAttr
from typing import List, Optional
from datetime import datetime, date
import pandas as pd
import enum
from loguru import logger
import sys

from llm.rag_utils import (
    get_documents_dataframe,
    get_tables_dataframe,
    get_tables_raw_dataframe,
)
from llm.utils import get_response_with_instructor


"""
Module: RAG Document Retrieval and Question Answering

This module implements a **Retrieval-Augmented Generation (RAG)** system for answering questions
by retrieving relevant documents and tables, analyzing them, and generating natural language answers.

The key components include:
1. `Retriever`: Handles document and table retrieval based on user questions and metadata.
2. `Generator`: Generates natural language answers from the retrieved context.
3. `ContextFormatter`: Formats retrieved tables and contexts for query answering.
4. `RAGAssistant`: An end-to-end assistant for answering user queries using the RAG system.

Dependencies:
- pandas (pd): For data table handling.
"""



# initialize logger
logger.remove()

logger.level("MAIN", no=35, color="<red>", icon="🔥")
logger.level("MINOR", no=25, color="<blue>", icon="🌊")
logger.level("OTHER", no=20, color="<green>")

# Add methods to logger
def main(message, *args, **kwargs):
    logger.log("MAIN", message, *args, **kwargs)

def minor(message, *args, **kwargs):
    logger.log("MINOR", message, *args, **kwargs)

def other(message, *args, **kwargs):
    logger.log("OTHER", message, *args, **kwargs)

logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>"
)

# Attach custom methods to the logger
logger.main = main
logger.minor = minor
logger.other = other


class Company(enum.Enum):
    """
    Represents a predefined list of companies that can be queried.

    Attributes:
        ENEA (str): Represents the ENEA company.
        TAURON (str): Represents the TAURON company.
        PKOBP (str): Represents the PKO BP bank.
        KGHM (str): Represents the KGHM company.
        PGE (str): Represents the PGE company.
        ORLEN (str): Represents the ORLEN company.
        PZU (str): Represents the PZU financial group.
    """
    ENEA = 1
    TAURON = 2
    PKOBP = 3
    KGHM = 4
    PGE = 5
    ORLEN = 6
    PZU = 7


company_descriptions = {
    Company.ENEA: "Enea SA to polska grupa energetyczna.",
    Company.TAURON: "Tauron Polska Energia S.A. to firma zajmująca się dystrybucją energii.",
    Company.PKOBP: "PKO Bank Polski (też znane jako PKO BP, PKO S.A.) to największy bank w Polsce.",
    Company.KGHM: "KGHM Polska Miedź S.A. to globalny producent miedzi.",
    Company.PGE: "PGE (lub Polska Grupa Energetyczna) to firma energetyczna.",
    Company.ORLEN: "Orlen PKN to koncern paliwowy.",
    Company.PZU: "PZU (lub Powszechny Zakład Ubezpieczeń) to wiodąca firma ubezpieczeniowa w Polsce."
}


class TimePeriod(BaseModel):
    """
    Represents a time period with start and end dates.

    Attributes:
        start_date (str): The start date of the period (ISO format).
        end_date (str): The end date of the period (ISO format).
    """
    start_date: date = Field(..., description="Data początkowa")
    end_date: date = Field(..., description="Data końcowa")


class Question(BaseModel):
    """
    Represents a user question with metadata about company and time period.

    Attributes:
        original_question (str): The original user question in natural language.
        company (str): Company name relevant to the question.
        time_period (TimePeriod): Time period associated with the user question.
    """
    original_question: str = Field(
        ...,
        description="Oryginalne pytanie zadane przez użytkownika"
    )
    company: List[Company] = Field(
        ...,
        description="""
            Lista firm, gdzie każda firma jest reprezentowana przez jeden z dostępnych obiektów enum.
            Domyślnie zwróć wszystkie.
            Przykłady: """ + ", ".join([f"{key.name}: {value}" for key, value in company_descriptions.items()]))
    time_period: TimePeriod = Field(
        ...,
        description="Zakres czasowy danych. Domyślnie poprzedni rok"
    )


class TableContext(BaseModel):
    """
    Encapsulates the context for a question, including reasoning and relevant tables.

    Attributes:
        question (Question): The question object for this context.
        reasoning_steps (str): Explanation of the reasoning for table relevance.
        explain_your_decision (str): Additional explanation for the decision-making process.
        relevant_table_ids (list): List of IDs for tables relevant to this context.
    """
    question: str = Field(..., description="Pytanie, na które chcemy uzyskać odpowiedź.")
    reasoning_steps: List[str] = Field(
        ...,
        description="""
            Lista rozważań lub kroków myślowych, które pomagają określić najlepsze tabele do odpowiedzi na pytanie.
        """
    )

    explain_your_decision: List[str] = Field(
        ...,
        description="Wyjaśnij swoja decyzje (co spowodowalo, ze wybierasz te tabele)"
    )

    relevant_table_ids: conlist(int, min_length=1, max_length=5) = Field(
        ...,
        description="""
            Lista identyfikatorów tabel, które są najbardziej odpowiednie do odpowiedzi na pytanie (od 1 do 5 tabel).
        """
    )


class TableContextWithExtendedAnalysis(BaseModel):
    user_question_relevant_table_ids: TableContext = Field(
        ...,
        description="""
            Lista identyfikatorów tabel, które są najbardziej odpowiednie do odpowiedzi na pytanie (od 1 do 5 tabel).
        """
    )

    follow_up_relevant_table_ids: List[TableContext] = Field(
        ...,
        description="""
            Lista identyfikatorów tabel, które są najbardziej odpowiednie do odpowiedzi na każde dodatkowe pytania
            (od 1 do 5 tabel).
        """
    )


###################################################################################################
#
#                    Retriever
#
###################################################################################################
class Retriever(BaseModel):
    """
    Handles retrieval of documents and tables related to a user's question.

    Attributes:
        user_question (Question): The user question to process.
        follow_up_questions (list): List of follow-up questions to refine the retrieval.
        _df_docs (pd.DataFrame): Private store of document metadata or content.
        _df_tables (pd.DataFrame): Private store of table metadata or content.
        _found_tables (list): Tables identified as relevant to the user's question.
        _follow_up_tables (list): Tables identified for follow-up queries.
        _user_question_tables (list): Tables directly relevant to the question.

    Methods:
        found_context_tables(): Finds context tables related to the question.
        search(): Searches for relevant documents and tables.
        _load_data(): Loads data into internal structures for retrieval.
        _get_company_documents(): Filters company-specific documents.
        _get_tables_from_documents(): Extracts tables from document metadata.
        _table_metadata(): Processes metadata for all tables.
        _generate_table_context(): Generates a context from retrieved tables.
        _find_relevant_tables(): Identifies tables most relevant to the question.
        _process_search_results(): Processes retrieved results for use.
        verbose(): Provides a detailed output of the retrieval process.
    """
    user_question: Question

    follow_up_questons: List[Question] = Field(
        ...,
        description="Jesteś ekspertem od analizy sprawozdań finansowych, specjalizującym się w identyfikacji kluczowych wskaźników i zrozumieniu struktury finansowej przedsiębiorstwa. Na podstawie dostarczonych danych finansowych (np. bilans, rachunek zysków i strat, przepływy pieniężne), Twoim zadaniem jest stworzenie do 5 pomocniczych pytań, które mają jednoznaczne odpowiedzi dostępne w tabelach zawartych w sprawozdaniu finansowym. Pytania powinny być konkretne, mierzalne i dotyczyć kluczowych aspektów analizy finansowej, takich jak rentowność, płynność, zadłużenie czy efektywność operacyjna, z uwzględnieniem informacji, które można bezpośrednio odczytać z dostarczonych danych."
    )

    _df_docs: pd.DataFrame = PrivateAttr(default=pd.DataFrame())
    _df_tables: pd.DataFrame = PrivateAttr(default=pd.DataFrame())
    _found_tables: pd.DataFrame = PrivateAttr(default=pd.DataFrame())
    _follow_up_tables: pd.DataFrame = PrivateAttr(default=pd.DataFrame())
    _user_question_tables: pd.DataFrame = PrivateAttr(default=pd.DataFrame())

    def found_context_tables(self, context: str):
        system_prompt = f"""
            Jesteś ekspertem w analizie sprawozdań finansowych. 
            Twoim zadaniem jest wybrać maksymalnie 5 tabel, które najlepiej odpowiadają na pytanie, korzystając z listy tabel
            zawierającej:
            - ID Tabeli: Unikalny identyfikator.
            - Tytuł: Nazwa tabeli wskazująca jej zawartość.

            Twoje zadanie:
            1. Przeanalizuj pytanie, identyfikując kluczowe elementy (np. zakres czasowy, rodzaj danych).
            2. Na podstawie tytułów wybierz najbardziej odpowiednie tabele.
            3. Uzasadnij każdy wybór.

            Tabele:
            {context}

        """

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Znajdź najbardziej pasujące tabele, w których są odpowiedzi na wszystkie pytania: {self}"},
        ]

        response = get_response_with_instructor(
            messages=messages,
            response_model=TableContextWithExtendedAnalysis,
            temperature=0,
            max_tokens=16000,
        )

        return response

    def search(self):
        logger.info("Run search")

        self._load_data()
        # Get documents related to companies in the user question
        doc_ids = self._get_company_documents()
        if not len(doc_ids):
            logger.warning("No documents found for the specified companies.")
            return None

        # Get tables from these documents
        self._get_tables_from_documents(doc_ids)
        if self._found_tables.empty:
            logger.warning("No tables found for the specified companies.")
            return None

        # Generate context information and find relevant tables
        context_info = self._generate_table_context()
        result = self._find_relevant_tables(context_info)

        # Process and store the search results
        self._process_search_results(result)

        return result

    def _load_data(self):
        self._df_docs = get_documents_dataframe()

        df_tables_raw = get_tables_raw_dataframe()
        df_tables_llm = get_tables_dataframe()

        self._df_tables = pd.merge(
            df_tables_llm,
            df_tables_raw[["document_id", "table_id", "markdown"]],
            on=["document_id", "table_id"]
        )

    def _get_company_documents(self):
        """Get document IDs related to companies in the user question."""
        company_ids = [c.value for c in self.user_question.company]
        doc_ids = self._df_docs[self._df_docs["company_id"].isin(company_ids)]["id"].values
        logger.info(f"Found {len(doc_ids)} documents")
        return doc_ids

    def _get_tables_from_documents(self, doc_ids):
        """Get tables from the specified documents."""
        self._found_tables = self._df_tables[self._df_tables["document_id"].isin(doc_ids)]
        logger.info(f"Found {len(self._found_tables)} tables in documents")

    def _table_metadata(self, row):
        line = f"table_id={row['table_id']}: "  # title={row['title']}, "
        line += f"description={row['description']}, "  # tags={row['tags']}, "
        line += f"practical_applications={row['practical_applications']}"

        return line

    def _generate_table_context(self):
        """Generate context information for the tables."""
        return "\n".join(self._found_tables.apply(self._table_metadata, axis=1).values)

    def _find_relevant_tables(self, context_info):
        """Use LLM to find the most relevant tables for the questions."""
        logger.info("Using LLM to find most relevant tables...")
        # found_context_tables is defined above
        return self.found_context_tables(context_info)

    def _process_search_results(self, result):
        """Process and store the search results."""
        # Handle user question tables
        user_question_tables = result.user_question_relevant_table_ids.relevant_table_ids
        logger.info(f"Most relevant tables for user question: {user_question_tables}")

        self._user_question_tables = self._found_tables[self._found_tables["table_id"].isin(user_question_tables)]

        for i, q_context in enumerate(result.follow_up_relevant_table_ids):
            q_table_ids = q_context.relevant_table_ids
            q_tables = self._found_tables[self._found_tables["table_id"].isin(q_table_ids)]
            self._follow_up_tables = pd.concat([self._follow_up_tables, q_tables], axis=0)

            logger.info(f"Follow-up question {i + 1}: {self.follow_up_questons[i].original_question}")
            logger.info(f"Relevant tables: {q_table_ids}")

        self._follow_up_tables = self._follow_up_tables.drop_duplicates(subset=["document_id", "table_id"])
        logger.info(f"Unique follow up tables: {self._follow_up_tables.shape[0]}")

    def verbose(self):
        logger.info("verbose")
        logger.main(self.user_question.original_question)
        logger.info("Firmy:")
        for c in self.user_question.company:
            logger.minor(c)
        logger.info("Zakres czasowy:")
        logger.minor(self.user_question.time_period.start_date)
        logger.minor(self.user_question.time_period.end_date)
        logger.info("Pomocnicze pytania:")
        for q in self.follow_up_questons:
            logger.minor(q.original_question)


def ask_fa(query):
    """
    Handles the logic to ask a specific question and interact with the RAG Assistant.

    Parameters:
        QUESTION (str): The natural language question to be asked.
        assistant (RAGAssistant): The RAG assistant object to process the question.
        result (str): Result message or placeholder.

    Returns:
        str: The answer produced by the assistant.
    """
    system_prompt = f"""
        Jesteś asystentem finansowym.

        Dzisiaj jest {date.today()}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    response = get_response_with_instructor(
        messages=messages,
        response_model=Retriever,
        temperature=0,
        max_tokens=2000,
    )

    return response


###################################################################################################
#
#                    Generator
#
###################################################################################################
class GeneratorAnswer(BaseModel):
    """
    Represents a generated answer for a given question.

    Attributes:
        question (Question): The user question being addressed.
        answer (str): The generated answer.
        explanation (str): A textual explanation of how the answer was generated.
    """
    question: str = Field(..., description="Treść pytania.")
    answer: str = Field(..., description="Odpowiedź na pytanie.")
    explanation: List[str] = Field(
        ...,
        description="Precyzyjne wyjaśnienie, dlaczego udzielono takiej odpowiedzi — co wpłynęło na taką decyzję."
    )


class ContextFormatter:
    """
    Formats the context and tables for generator input.

    Methods:
        format_row(): Formats a single row of a table for easier comprehension.
        format_context(): Formats the entire context for generation or debugging.
    """
    @staticmethod
    def format_row(row):
        return f"""
            Description: {row['description']}
            Tags: {", ".join(row['tags'])}

            {row['markdown']}
            ---------------------------------------------------
        """

    @staticmethod
    def format_context(retriever):
        follow_up_questions = "\n".join([q.original_question for q in retriever.follow_up_questons])
        context_with_relevant_data = "".join(
            retriever._follow_up_tables.apply(ContextFormatter.format_row, axis=1).values
        )

        return follow_up_questions, context_with_relevant_data


class Generator:
    """
    Handles the generation of natural language answers from table-based context.

    Methods:
        generate(): Creates a natural language answer based on context and user question.
    """
    def generate(self, retriever) -> GeneratorAnswer:
        follow_up_questions, context_with_relevant_data = ContextFormatter.format_context(retriever)

        system_prompt = f"""
            Odpowiedź na pytanie użytkownika na podstawie kontekstu.

            POMOCNICZE PYTANIA (które warto wziąć pod uwagę):
            {follow_up_questions}

            KONTEKST:
            {context_with_relevant_data}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": retriever.user_question.original_question},
        ]

        response = get_response_with_instructor(
            messages=messages,
            response_model=GeneratorAnswer,
            temperature=0,
            max_tokens=2000,
        )

        return response


###################################################################################################
#
#                    RAGAssistant
#
###################################################################################################

class RAGAssistant:
    """
    End-to-end Retrieval-Augmented Generation assistant for question answering.

    Attributes:
        logger (Logger): Logger instance for debugging and tracking the process.
        generator (Generator): The generator instance for producing answers.

    Methods:
        __init__(): Initializes the RAG Assistant with necessary dependencies.
        ask(): Handles the end-to-end process of retrieval and answer generation for a user query.
    """
    def __init__(self, logger=None):
        self.generator = Generator()
        self.logger = logger

    def ask(self, question: str) -> GeneratorAnswer:
        """Przetwarzanie pytania od początku do końca"""
        # Wykorzystanie istniejącej funkcji ask_fa do utworzenia Retrievera
        retriever = ask_fa(question)
        retriever.search()

        # Generowanie odpowiedzi
        answer = self.generator.generate(retriever)

        # Logowanie, jeśli logger jest dostępny
        if self.logger:
            self.logger.main(answer.question)
            self.logger.other(answer.answer)
            self.logger.info(answer.explanation)

        return answer


if __name__ == "__main__":
    """
    Main function to execute the RAG question-answering system.

    Orchestrates the retrieval, context formatting, and generation steps to provide an answer
    for a predefined question or set of questions.
    """
    QUESTION = "Jak wygląda struktura kapitałowa i jaki jest stosunek kapitału własnego do obcego w PKOBP?"

    assistant = RAGAssistant(logger=logger)
    result = assistant.ask(QUESTION)
