'''
🧠 Zadanie 1.2.1
Twoim zadaniem jest stworzenie prostego asystenta, który będzie działał w określonym kontekście i
zgodnie z wybranymi przez Ciebie założeniami. Możesz nadać mu imię, określić sposób, w jaki
powinien się komunikować (np. formalnie lub nieformalnie), oraz zaprojektować jego "osobowość".
Dodatkowo spróbuj wyznaczyć mu konkretne zastosowanie – na przykład asystent może pomagać
w tłumaczeniach, udzielać wskazówek w wybranej dziedzinie, czy wspierać użytkownika w codziennych
zadaniach.

Celem jest zarówno określenie "zachowania" asystenta, jak i zaproponowanie praktycznego
scenariusza, w którym jego działanie przyniesie wartość użytkownikowi. Pomyśl kreatywnie i
postaraj się zaprojektować coś, co mogłoby być naprawdę użyteczne i inspirujące!
'''


from llm.utils import get_simple_response

def books_assistant(user_prompt):
    role_system_desc = '''
        You are an expert who recommends books similar to book (author and title) from a query.
        Your task is to recommend 3 books written by 3 different authors (exclude the author from query).
        The response should include: author/authors, title and short summary (max 2 sentences).
        The answer for each position should 2 lines: author and title in the first line and summary in the second line.
        Start from the most relevant position. Add numeration.
    '''
    messages = [
        { "role": "system", "content": role_system_desc },
        { "role": "user", "content": user_prompt}
    ]

    return get_simple_response(messages)


if __name__ == "__main__":
    response = books_assistant("Carl Sagan, Pale Blue Dot")
    print(response)
    response = books_assistant("Michio Kaku, Quantum supremacy")
    print(response)