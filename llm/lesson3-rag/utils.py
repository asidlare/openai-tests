import os


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def read_markdown(file_name):
    full_path = f"{root_path}/llm/lesson3-rag/md/{file_name}"
    with open(full_path) as f:
        return "".join( f.readlines() )


def extract_table(section):
    tables = []

    prev_line = ""
    table_start = False
    current_table = []

    for line in section.split("\n"):
        if "|---|" in line:
            table_start = True
            current_table.append(prev_line)
        elif ("---" in line) or (line.strip() == ""):
            table_start = False
            if current_table:
                table_text = "\n".join(current_table)
                tables.append(table_text)

            current_table = []

        if table_start:
            current_table.append(line)

        prev_line = line

    return tables


def extract_tables(document):
    content = read_markdown(document)
    sections = content.split("# ")

    tables = []
    for section in sections:
        tables += extract_table(section)

    return tables


documents = {
    'PGE': 'PGE_Jednostkowe-sprawozdanie-finansowe-PGE-za-rok-2023-plik-pdf-nie-stanowiacy-wersji-oficjalnej.pdf.md',
    'PZU': 'PZU_Sprawozdanie_finansowe_PZU_SA_2023.pdf.md',
    'KGHM': 'Sprawozdanie finansowe KGHM RR_2023.xhtml.pdf.md',
    'TAURON': 'TAURON-SprFinan-2023-12-31-PL.pdf.md',
    'ENEA': 'enea_jednostkowe_sprawozdanie_finansowe_enea_s_a_.pdf.md',
    'ORLEN': 'orlen_4_SPRAWOZDANIEFINANSOWEORLEN_2023.xhtml.pdf.md',
    'PKOBP': 'pkobp_Jednostkowe_sprawozdanie_finansowe_PKO_Banku_Polskiego_S.A._za_2023_rok_pdefowe.pdf.md',
}