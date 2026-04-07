from pathlib import Path
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: Path) -> str:
    # abre o arquivo PDF a partir do caminho informado e prepara o leitor para percorrer as páginas
    reader = PdfReader(str(pdf_path))

    # lista que vai acumulando o texto extraído de cada página separadamente
    full_text = []

    for page in reader.pages:
        # tenta extrair o texto da página atual (pode retornar None em páginas só com imagens)
        text = page.extract_text()

        # só adiciona se realmente veio algum conteúdo (páginas em branco ou com só imagens retornam None)
        if text:
            full_text.append(text)

    # junta todas as páginas em uma única string, separando cada uma por uma linha em branco
    return "\n".join(full_text)