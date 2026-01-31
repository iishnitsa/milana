'''
search_web
send 1 search query after the command
Simple web search
Uses AI to summarize the results
'''

import re
import requests
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse, urljoin
from ddgs import DDGS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from cross_gpt import let_log, text_cutter, cacher

# Улучшенные заголовки для имитации реального браузера
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
}

# Дополнительные User-Agent для ротации
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0'
]

LINKS_PER_ENGINE = 10

def requests_retry_session(
    retries=3,
    backoff_factor=0.5,
    status_forcelist=(500, 502, 504, 403),
    session=None,
):
    """Creates a requests session with retry logic"""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods={"GET", "POST"},
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

@cacher
def fetch_links_ddg(query, max_results=10):
    results = []
    try:
        with DDGS() as ddgs: # Use generator to get results
            for r in ddgs.text(query, max_results=max_results):
                results.append(r['href'])
                if len(results) >= max_results: break
    except Exception as e: let_log(f'[fetch_links] DDGS error: {str(e)}')
    return results if results else []

@cacher
def get_page_text(url):
    try:
        # Random delay before request
        time.sleep(random.uniform(1, 3))
        # Create session with retry logic
        session = requests_retry_session(retries=2, status_forcelist=(403, 500, 502, 504))
        # Random User-Agent selection
        headers = HEADERS.copy()
        headers['User-Agent'] = random.choice(USER_AGENTS)
        # UNIVERSAL REFERER LOGIC
        try:
            parsed_url = urlparse(url)
            # Set Referer as the site's base URL
            headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}/"
        except Exception as e:
            let_log(f'[get_page_text] Error parsing URL {url}: {str(e)}')
            # Fallback to Google if URL parsing fails
            headers['Referer'] = 'https://www.google.com/'
        # Make request with increased timeout
        r = session.get(
            url,
            headers=headers,
            timeout=(10, 30)  # (connect timeout, read timeout)
        )
        # Special handling for 403
        if r.status_code == 403:
            let_log(f'[get_page_text] Access forbidden (403) for {url}. Trying alternative approach...')
            # Try another User-Agent
            headers['User-Agent'] = random.choice(USER_AGENTS)
            r = session.get(url, headers=headers, timeout=(10, 30))
            let_log(f'[get_page_text] Retry request, status: {r.status_code}')
        if not r.ok:
            let_log(f'[get_page_text] HTTP error: {r.status_code}')
            return ''
        content_type = r.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            let_log(f'[get_page_text] Unsupported content type: {content_type}')
            return ''
        
        # Пункт 3: Установите правильную кодировку
        try:
            # Автоопределение кодировки
            r.encoding = r.apparent_encoding
            let_log(f'[get_page_text] Detected encoding for {url}: {r.encoding}')
        except Exception as e:
            let_log(f'[get_page_text] Error detecting encoding for {url}: {str(e)}')
            # Fallback к UTF-8 если не удалось определить
            r.encoding = 'utf-8'
        
        soup = BeautifulSoup(r.text, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]): script.decompose()
        # Additional cleanup
        for element in soup(["nav", "footer", "header", "aside", "form"]): element.decompose()
        
        # Пункт 2: Улучшенный поиск основного контента
        selectors = [
            'main', 'article', 
            '[role="main"]', 
            '.main-content', '.article-content',
            '.post-content', '.entry-content',
            '#content', '.content',
            '#main', '.main',
            '.story', '.story-content',
            '#article', '.article'
        ]
        
        main_content = None
        for selector in selectors:
            try:
                found = soup.select_one(selector)
                if found and len(found.get_text(strip=True)) > 100:  # Проверяем, что есть достаточный текст
                    main_content = found
                    let_log(f'[get_page_text] Found content with selector: {selector}')
                    break
            except Exception as e:
                continue  # Пропускаем некорректные селекторы
        
        if main_content: 
            text = main_content.get_text(" ", strip=True)
        else: 
            # Если не нашли по селекторам, попробуем эвристический подход
            # Ищем элемент с наибольшим количеством текста
            all_elements = soup.find_all(['div', 'section'])
            best_element = None
            max_text_length = 0
            
            for element in all_elements:
                element_text = element.get_text(strip=True)
                if len(element_text) > max_text_length and len(element_text) > 200:
                    max_text_length = len(element_text)
                    best_element = element
            
            if best_element:
                text = best_element.get_text(" ", strip=True)
                let_log(f'[get_page_text] Using heuristic approach, found element with {max_text_length} chars')
            else:
                text = soup.get_text(" ", strip=True)
                let_log(f'[get_page_text] Using full page text')
        
        # Remove extra spaces and line breaks
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Пункт 1: Добавьте проверку длины текста
        if len(text) < 100:  # Минимальная длина текста увеличена до 100 символов
            let_log(f'[get_page_text] Text too short ({len(text)} chars), returning empty')
            return ''
            
        # Дополнительная проверка на качество контента
        # Проверяем соотношение полезного текста к общему количеству символов
        words = text.split()
        if len(words) < 20:  # Меньше 20 слов - вероятно, не статья
            let_log(f'[get_page_text] Too few words ({len(words)}), returning empty')
            return ''
        
        # Проверка на наличие "мусорного" контента (например, только навигация)
        if any(bad_text in text.lower() for bad_text in ['404', 'not found', 'page not found', 'access denied']):
            let_log(f'[get_page_text] Bad content detected, returning empty')
            return ''
            
        let_log(f'[get_page_text] Successfully extracted {len(text)} chars from {url}')
        return text
        
    except Exception as e:
        let_log(f'[get_page_text] Error processing page {url}: {str(e)}')
        return ''

def main(text):
    if not hasattr(main, 'attr_names'):
        let_log('INITIALIZATION')
        main.attr_names = (
            'search_failed',
            'results_prefix',
            'search_error_msg',
            'no_results_msg'
        )
        main.search_failed = 'No text found on visited pages'
        main.results_prefix = 'Here\'s what I found\n'
        main.search_error_msg = 'Search error: '
        main.no_results_msg = 'No results found. Try refining your query or using different keywords.'
        return
        
    let_log('WEB SEARCH CALLED')
    all_links = []
    
    try:
        links = fetch_links_ddg(text, LINKS_PER_ENGINE)
        all_links.extend(links)
    except Exception as e: return f'{main.search_error_msg}{str(e)}'

    # Remove duplicates while preserving order
    unique_links = list(dict.fromkeys(all_links))
    # If no links found
    if not unique_links: return main.no_results_msg

    # Process all found links (no page limit)
    collected_text = ""
    for i, link in enumerate(unique_links):
        page_text = get_page_text(link)
        if page_text: collected_text += f"=== Source: {link} ===\n{page_text}\n\n"
        # Add random delay between pages
        if i < len(unique_links) - 1:
            time.sleep(random.uniform(2, 4))

    if not collected_text.strip(): return main.search_failed
    return main.results_prefix + text_cutter(collected_text)
