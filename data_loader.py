from typing import Optional, Dict, List
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

def get_session() -> Optional[requests.Session]:
    login_url = "https://help.myagent.online/how-to/aviabiletyi/faq-po-aviabiletam/"

    payload = {
        "username": "help",
        "password": "agentagent",
        "returnUrl": "/how-to/aviabiletyi/faq-po-aviabiletam/",
        "service": "login"
    }

    session = requests.Session()

    response = session.post(login_url, data=payload)
    if "error" in response.text.lower():
        print("Ошибка авторизации!")
        return None
    print("Успешный вход!")
    return session

def fetch_content_main_page() -> Optional[List[Document]]:

    def fetch_content_url(url: str) -> Optional[str]:
        response = session.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        main_content = soup.find('div', id='mainer')

        menu_div = main_content.find('div', class_='menu-on-page')
        if menu_div:
            menu_links = menu_div.find_all('a')
            for link in menu_links:
                full_link = "https://help.myagent.online/" + link['href']
                if full_link not in main_links:
                    main_links.append(full_link)

        text = [f"# {soup.find('title').get_text(strip=True)}"]

        for element in main_content.find_all(['ul', 'p', 'h1', 'h2', 'h3', 'h4', 'h5'], recursive=True):

            if element.name == 'ul' and 'list-inline' in element.get('class', []):
                break

            if element.name == 'p':
                content = element.get_text(strip=True)
                if content:
                    text.append(content)

            elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5']:
                content = element.get_text(strip=True)
                if content:
                    text.append(f"## {content}")

            elif element.name == 'ul':
                for li in element.find_all('li'):
                    content = li.get_text(strip=True)
                    if content:
                        text.append(f"- {content}")
        if len(text) == 1:
            return None
        return '\n'.join(text)

    main_url = "https://help.myagent.online/how-to/"
    session = get_session()
    if not session:
        print('Не удалось получить данные!')
        return None
    main_response = session.get(main_url)
    main_soup = BeautifulSoup(main_response.content, 'html.parser')
    linker_div = main_soup.find('div', class_='linker')
    main_links = linker_div.find_all('a')
    main_links = ["https://help.myagent.online/" + link['href'] for link in main_links]

    documents = []
    for main_link in main_links:
        page_content = fetch_content_url(main_link)
        if page_content:
            documents.append(Document(page_content=page_content))
    print(f"Создано {len(documents)} документов")
    return documents

if __name__ == "__main__":
    fetch_content_main_page()