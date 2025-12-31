'''
search_web
send a search query after the command
Simple web search
Uses AI to summarize the results
'''

import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse, urljoin
from cross_gpt import ask_model, text_cutter, let_log, cacher

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; agent-console/1.0)'
}

LINKS_PER_ENGINE = 10
MAX_RECURSION_DEPTH = 1

SEARCH_ENGINES = {
    'duckduckgo': {
        'url':  'https://html.duckduckgo.com/html?q={q}&s={off}',
        'per_page': 30,
        'offset_step': 30,
        'parser': lambda soup: [a['href'] for a in soup.select('a.result__a')],
    },
    'bing': {
        'url':  'https://www.bing.com/search?q={q}&first={off}',
        'per_page': 10,
        'offset_step': 10,
        'parser': lambda soup: [a['href'] for a in soup.select('li.b_algo h2 > a')],
    },
    'startpage': {
        'url':  'https://www.startpage.com/sp/search?query={q}&page={page}',
        'per_page': 10,
        'offset_step': 1,
        'parser': lambda soup: [a['href'] for a in soup.select('a.result-link')],
    },
    'qwant': {
        'url': 'https://www.qwant.com/?q={q}&t=web&p={page}',
        'per_page': 10,
        'offset_step': 1,
        'parser': lambda soup: [a['href'] for a in soup.select('a[href^="http"]')],
    }
}

def fetch_links(engine, config, query, needed):
    results = []
    offset = 0
    page = 1
    while len(results) < needed:
        try:
            if 'page' in config['url']:
                url = config['url'].format(q=quote_plus(query), page=page)
                page += config['offset_step']
            else:
                url = config['url'].format(q=quote_plus(query), off=offset)
                offset += config['offset_step']
            r = cacher()
            if r is None:
                r = requests.get(url, headers=HEADERS, timeout=10)
                cacher(r)
            soup = BeautifulSoup(r.text, 'html.parser')
            found = config['parser'](soup)
            results.extend(found)
            if not found:
                break
            time.sleep(1)
        except Exception:
            break
    return results[:needed]


def extract_internal_links(html: str, base_url: str, limit=2):
    soup = BeautifulSoup(html, 'html.parser')
    base_domain = urlparse(base_url).netloc
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(base_url, href)
        if urlparse(full_url).netloc == base_domain:
            links.append(full_url)
        if len(links) >= limit:
            break
    return links


def get_page_text_recursive(url, depth=0, max_depth=1, visited=None):
    if visited is None:
        visited = set()
    if url in visited or depth > max_depth:
        return ''
    visited.add(url)

    try:
        r = cacher()
        if r is None:
            r = requests.get(url, headers=HEADERS, timeout=10)
            cacher(r)
        r = requests.get(url, headers=HEADERS, timeout=10)
        if not r.ok or 'text/html' not in r.headers.get('Content-Type', ''):
            return ''
        soup = BeautifulSoup(r.text, 'html.parser')
        text = soup.get_text(" ", strip=True)
    except Exception:
        return ''

    links = extract_internal_links(r.text, url, limit=2)
    child_texts = [
        get_page_text_recursive(link, depth + 1, max_depth, visited)
        for link in links
    ]
    return text + '\n' + '\n'.join(child_texts)


def main(text):
    if not hasattr(main, 'attr_names'):
        let_log('ИНИЦИАЛИЗАЦИЯ')
        main.attr_names = ('search_failed', 'summarizing_failed', 'summarize_prefix')
        main.search_failed = 'No text found on visited pages'
        main.summarizing_failed = 'Failed to summarize the content'
        main.summarize_prefix = 'Summarize the following:\n'
        return
    let_log('ВЕБ ПОИСК ВЫЗВАН')

    all_links = []
    for name, config in SEARCH_ENGINES.items():
        try:
            links = fetch_links(name, config, text, LINKS_PER_ENGINE)
            all_links.extend(links)
        except Exception:
            continue

    unique_links = list(dict.fromkeys(all_links))  # preserve order

    collected_text = ""
    for link in unique_links:
        page_text = get_page_text_recursive(link, max_depth=MAX_RECURSION_DEPTH)
        if page_text:
            collected_text += page_text + "\n"

    if not collected_text.strip():
        return main.search_failed

    try:
        return ask_model(f"{main.summarize_prefix}{text_cutter(collected_text)}", all_user=True)
    except Exception:
        return ask_model(f"{main.summarize_prefix}{text_cutter(collected_text)}", all_user=True)