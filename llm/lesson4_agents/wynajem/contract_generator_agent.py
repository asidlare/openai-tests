from datetime import date
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from llm.lesson4_agents.wynajem.base_agent import (
    BaseAgent,
    Party,
)


class ClauseTemplate(BaseModel):
    """
    Reprezentuje szablon danych wymaganych do generowania klauzuli.
    """
    required_fields: Dict[str, str] = Field(
        ...,
        description="Wymagane pola i ich opis (nazwa: opis z przykładem)."
    )
    optional_fields: Optional[Dict[str, str]] = Field(
        None,
        description="Opcjonalne pola i ich opis (nazwa: opis)."
    )


class Clause(BaseModel):
    """
    Reprezentuje pojedynczą klauzulę w paragrafie
    """

    id: str = Field(..., description="Unikalny identyfikator klauzuli, np. '1.1'.")

    chain_of_thought: List[str] = Field(
        ...,
        description="""
        Kroki myślowe prowadzące do stworzenia klauzuli. Każdy krok powinien odpowiadać pytaniu lub decyzji, np.:
        1. Jakie prawo, obowiązek lub warunek musi zostać określony?
        2. Jakie potencjalne wyjątki lub ograniczenia mogą mieć zastosowanie?
        3. Czy istnieją powiązania z innymi klauzulami w dokumencie?
        4. Czy sformułowanie jest zgodne z przepisami prawa?
        5. Czy język klauzuli jest dostosowany do formalnych wymogów?
        """
    )

    template: Optional[ClauseTemplate] = Field(
        None,
        description="Definicja danych wymaganych i opcjonalnych dla klauzuli."
    )

    note: Optional[str] = Field(None, description="Dodatkowy komentarz lub uwaga do klauzuli.")

    text: str = Field(
        ...,
        description="""
        Treść klauzuli sformułowana zgodnie z zasadami języka prawnego. Musi spełniać następujące kryteria:

        1. Konkretność:
        - jednoznaczne określenie praw, obowiązków lub warunków
        - brak niejasności i wieloznaczności w zapisach

        2. Formalność:
        - stosowanie stylu urzędowego
        - wykluczenie języka potocznego i kolokwializmów

        3. Spójność:
        - logiczne powiązanie z pozostałymi klauzulami
        - brak sprzeczności z innymi postanowieniami

        4. Jednoznaczność:
        - zrozumiałość w kontekście prawnym
        - precyzyjne określenie warunków i zakresu stosowania

        5. Formatowanie:
        - dopuszczalny podział na zdania lub punkty
        - wyodrębnienie istotnych elementów (wyjątków, warunków, terminów)
        - stosowanie odpowiednich konstrukcji składniowych (np. 'z zastrzeżeniem, że')

        Przykład poprawnego zapisu:
        'Strona zobowiązuje się do dokonania płatności w terminie 14 dni od dnia otrzymania faktury, 
        z zastrzeżeniem, że w przypadku opóźnienia płatności naliczane będą odsetki ustawowe 
        w wysokości określonej w obowiązujących przepisach prawa.'
        """
    )


class Preamble(BaseModel):
    """
    Reprezentuje preambułę umowy, zawierającą dane stron i inne informacje wprowadzające.
    """
    contract_date: date = Field(..., description="Data zawarcia umowy")
    location: str = Field(..., description="Miejscowość, w której została zawarta umowa.")
    lessor: Party = Field(..., description="Dane wynajmującego czyli dane pierwszej strony umowy")
    lessee: Party = Field(..., description="Dane najemcy czyli dane drugiej strony umowy")


class Paragraph(BaseModel):
    """
    Reprezentuje pojedyńczy paragraf w umowie.
    """
    id: str = Field(..., description="Unikalny identyfikator paragrafu, np. '§1'.")
    goal: str = Field(None, description="Cel lub intencja paragrafu.")
    chain_of_thought: List[str] = Field(None, description="Kolejne kroki logiczne prowadzące do paragrafu.")
    title: str = Field(None, description="Opcjonalny tytuł paragrafu.")
    note: Optional[str] = Field(None, description="Dodatkowy komentarz lub uwaga do paragrafu.")
    clauses: List[Clause] = Field(..., description="Lista klauzul w paragrafie.")


class PartContract(BaseModel):
    """
    Reprezentuje część umowy składającą się z paragrafów.
    """
    paragraphs: List[Paragraph] = Field(..., description="Lista paragrafów w umowie.")


class Contract(BaseModel):
    """
    Reprezentuje całą umowę składającą się z paragrafów.
    """
    title: str = Field(..., description="Tytuł umowy, np. 'Umowa najmu'.")
    preamble: Preamble = Field(..., description="Preambuła umowy zawierająca wstępne dane i informacje.")
    paragraphs: List[Paragraph] = Field(..., description="Lista paragrafów w umowie.")
    version: Optional[str] = Field(None, description="Opcjonalna wersja umowy.")


CONTRACT_KNOWLEDGE_MAP = """
## §1 Strony umowy
- Kto jest Wynajmującym i Najemcą (pełne dane osobowe/firma)?
- Czy podane są dokładne adresy zamieszkania/siedziby stron?
- Czy wskazano podstawę działania stron (np. pełnomocnictwo)?

## §2 Przedmiot najmu
- Czy dokładnie opisano lokal (adres, powierzchnia, stan techniczny, przeznaczenie)?
- Czy uwzględniono wyposażenie lokalu?
- Czy załączono protokół zdawczo-odbiorczy jako załącznik?

## §3 Przeznaczenie lokalu
- Czy jasno określono cel najmu (mieszkalny, biurowy, inny)?
- Czy zawarto zakaz zmiany przeznaczenia bez zgody Wynajmującego?
- Czy wskazano, jakie rodzaje działalności są niedozwolone?

## §4 Okres obowiązywania umowy
- Czy określono, czy umowa jest na czas określony/nieokreślony i na jaki okres czasu?
- Czy wskazano dokładną datę rozpoczęcia i (jeśli dotyczy) zakończenia umowy?
- Czy wskazano, że przedłużonie umowy będzie wymagało osobnego aneksu?
- Czy wskazano wyraźnie, że umowa najmu zawarta została na piśmie, aby zapewnić zgodność z polskim prawem?
- Czy umową zawiera klauzulę dotyczącą terminu i sposobu wypowiedzenia najmu?
- Czy zawiera klauzulę dotyczącą możliwości wcześniejszego wypowiedzenia umowy przed upływem terminu na jaki została zawarta?
- Czy wskazano, że najemca obejrzał lokal i nie stwierdzono poważnych wad lokalu?

## §5 Czynsz i warunki płatności
- Czy określono wysokość czynszu oraz precyzyjnie określono terminy i formę płatności i konsekwencje ich niedotrzymania?
- Jakie są akceptowalne formy płatności (np. przelew, gotówka)?
- Czy uwzględniono ewentualne podwyżki czynszu i szczegółowe zapisy dotyczące procedury podwyżki czynszu?
- Czy umowa zawiera klauzulę dotyczącą wysokości czynszu?
- Czy umowa zawiera klauzulę dotyczącą możliwości rozwiązania umowy ze skutkiem natychmiastowym za opóźnienia w płatności conajmniej miesiąc?
- Czy wskazano, że czynsz pozostaje stały w trakcie obowiązywania umowy, a ewentualne zmiany będą wymagały aneksu?
- Czy zostały określone zasady podwyżki czynszu (np. z miesięcznym wyprzedzeniem), uwzględnione, że będą one wymagały aneksu oraz uwzględniono warunki wypowiedzenia przez wynajmującego?
- Czy określono, że koszty mediów to opłaty za prąd, wodę, ogrzewanie i precyzuje, że są one zależne od rachunków od dostawcy prądu?
- Czy określono, że koszty mediów nie wchodzą w skład czynszu i opłacane są oddzielnie zgodnie z aktualnym taryfami dostawcy?

## §6 Dodatkowe opłaty i obowiązki Najemcy
- Jakie opłaty dodatkowe obciążają Najemcę (media, eksploatacja, inne)?
- Czy wskazano obowiązki Najemcy dotyczące utrzymania lokalu i że to on ponosi ich koszty?
- Czy wyszczególniono zakres drobnych napraw, które są obowiązkiem wynajmującego zgodnie z Art. 681 kodeksu cywilnego?
- Czy dołączono klauzulę zawierającą treść Art. 681. kodeksu cywilnego, aby doprecyzować termin "drobne naprawy"?
- Czy opisano procedurę zgłaszania usterek?
- Czy zawiera klauzulę dotyczącą odwiedzialności wynajmującego za naprawy wykraczające poza art 681 kodeksu cywilnego?

##  §7 Kaucja zabezpieczająca
- Czy określono wysokość kaucji i sposób jej przekazania?
- Czy jasno i precyzyjnie określono zasady i sposób zwrotu kaucji po zakończeniu umowy?
- Czy wskazano, na co kaucja może zostać przeznaczona w przypadku szkód?
- Czy zawiera klauzulę dotyczącą zasad potrącenia kosztów napraw z kaucji?
- Czy umowa zawiera szczegółowe zapisy dotyczące warunków zwrotu kaucji
- Czy umowa zawiera szczegółową procedurę zwrotu kaucji i zasad potrącania kosztów przez najemcę po opuszczaniu lokalu przez wynajmującego?

## §8 Mieszkańcy lokalu
- Czy określono liczbę osób mogących mieszkać w lokalu?
- Czy wskazano możliwość przyjmowania innych osób na stałe?
- Jakie są zasady informowania Wynajmującego o zmianach w liczbie mieszkańców?
- Czy umowa zawiera klauzulę zabraniającą podnajmu lokalu i możliwość wypowiedzenia umowy ze skutkiem natychmiastowym w przypadku stwierdzenia podnajmu lokalu?

## §9 Ubezpieczenie lokalu
- Czy jasno określono, kto odpowiada za ubezpieczenie nieruchomości?
- Czy wskazano, czy Najemca jest zobowiązany do wykupienia polisy OC?
- Czy określono jakie szkody mają być objęte polisą OC?
- Czy zawiera klauzulę dotyczącą odpowiedzialności najemcy w przypadku braku ubezpieczenia lokalu?
- Czy zawiera klauzulę o pełnej odpowiedzialności najemcy za szkody w lokalu w przypadku braku wykupienia przez niego ubezpieczenia lokalu?

## §10 Zmiany w umowie
- Czy określono zasady wprowadzania zmian w umowie (forma pisemna)?
- Czy wskazano, kto ponosi koszty sporządzenia aneksu?

## §11 Odbiór i zwrot lokalu
- Czy opisano sposób odbioru lokalu (protokół zdawczo-odbiorczy)?
- Jakie są zasady zwrotu lokalu po zakończeniu najmu (termin, stan techniczny)?
- Czy wskazano szczegółowo konsekwencje braku zwrotu w ustalonym terminie?

## §12 Rozwiązanie umowy
- Jakie są warunki wypowiedzenia umowy przez każdą ze stron?
- Czy określono okres wypowiedzenia?
- Czy umowa zawiera klauzulę dotyczącą możliwości wypowiedzenia najmu przez każdą ze stron?
- Czy wskazano procedurę rozstrzygania spornych przypadków rozwiązania umowy?

## §13 Załączniki do umowy
- Czy wymieniono wszystkie załączniki (protokół zdawczo-odbiorczy, wykaz wyposażenia)?
- Czy wskazano, że załączniki stanowią integralną część umowy?

## §14 Postanowienia końcowe
- Czy określono, które przepisy prawa regulują umowę (np. Kodeks cywilny)?
- Czy uwzględniono klauzulę dotycząca rozstrzygania sporów (sąd właściwy)?
- Czy strony potwierdziły zapoznanie się z umową i jej załącznikami?

"""


class ContractGeneratorAgent(BaseAgent):
    @staticmethod
    def contract_parts(text, max_elementy, sep="## "):
        lst = text.split(sep)[1:]
        return [sep + sep.join(lst[i:i + max_elementy]) for i in range(0, len(lst), max_elementy)]

    def run_part(self, paragraph, response_model):
        system_prompt = f"""

                Jesteś ekspertem prawa cywilnego w Polsce, specjalizującym się w umowach najmu. Twoje zadanie polega na szczegółowym opracowaniu wskazanych paragrafów umowy najmu, zgodnie z aktualnym stanem prawnym i najlepszymi praktykami.

                Podczas generowania treści:
                1. Każdy paragraf rozpocznij od numeru i tytułu
                2. Używaj precyzyjnego języka prawniczego
                3. Dbaj o kompletność regulacji
                4. Uwzględniaj ochronę interesów obu stron
                5. Zapewnij zgodność z Kodeksem Cywilnym i ustawą o ochronie praw lokatorów

                Struktura każdego paragrafu powinna zawierać:
                - Postanowienia ogólne
                - Szczegółowe prawa i obowiązki stron
                - Konsekwencje naruszenia postanowień
                - Warunki szczególne (jeśli dotyczy)

                Wygeneruj treść wskazanych paragrafów. Obecnie skupiamy się na paragrafach:
                {paragraph}
                
                Dane do umowy:
                {self.context.contract_data.model_dump_json()}
                
                Dzisiejsza data: {date.today()}
                Data początku umowy: 1-szy dzień miesiąca następującego po dzisiejszej dacie.

                Zadbaj o spójność między paragrafami i zgodność z przepisami prawa.
                """

        try:
            result = self._get_api_call(
                response_model=response_model,
                system_prompt=system_prompt,
            )
            self.logger.info("[AgentGenerator]: Finished")
            return result
        except Exception as e:
            self.logger.error(f"Error! {e}")

    def run(self) -> bool:
        self.logger.info("[ContractGeneratorAgent] Generating contract...")
        paragraphs = []
        paragraphs.append('Umowa najmu')
        for part in self.contract_parts(CONTRACT_KNOWLEDGE_MAP, max_elementy=1):
            self.logger.info(part)

            retry = 0
            while True:
                if retry > 3:
                    self.logger.error(f"{part} - retry limit reached")
                    break
                result = self.run_part(part, Contract)

                if not result or getattr(result, 'paragraphs') is None:
                    self.logger.info(f"Attempt {retry} failed. Retrying...")
                    retry += 1
                    continue
                current_clauses = [f"{c.id} {c.text}" for p in result.paragraphs for c in p.clauses]
                paragraphs.extend(current_clauses)
                self.logger.info(current_clauses)
                break
        self.context.contract_text = "\n".join(paragraphs)
        self.logger.info(self.context.contract_text)
        return True
