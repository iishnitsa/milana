'''
web_search_links
performs DuckDuckGo internet search and returns links with titles and descriptions
Search URLs
performs DuckDuckGo internet search and returns links with titles and descriptions
'''

from ddgs import DDGS
from cross_gpt import let_log, found_info_1

def main(text):
    if not hasattr(main, 'attr_names'):
        let_log('INITIALIZATION')
        main.attr_names = (
            'search_error_msg',
            'no_title',
            'no_description',
            'title_text',
            'description_text',
            'search_error_general'
        )
        main.search_error_msg = 'Search error: '
        main.no_title = 'No title'
        main.no_description = 'No description'
        main.title_text = 'Title: '
        main.description_text = '\nDescription: '
        main.search_error_general = 'Search error'
        return
    let_log('WEB SEARCH LINKS CALLED')
    let_log(f'Search query: {text}')

    try:
        results = []
        # Добавлен таймаут 60 секунд, явно указан backend="auto"
        with DDGS(timeout=60) as ddgs:
            for r in ddgs.text(text, max_results=10, backend="auto"):
                title = r.get('title', main.no_title)
                body = r.get('body', main.no_description)
                href = r.get('href', '')
                results.append((title, body, href))
        if not results:
            return found_info_1
        output = []
        for title, body, href in results:
            output.append(f"{main.title_text}{title}{main.description_text}{body}\nURL: {href}\n")
        return '\n'.join(output)
    except Exception as e:
        let_log(f'Error in web_search_links: {str(e)}')
        # Возвращаем короткое сообщение без деталей ошибки
        return main.search_error_general