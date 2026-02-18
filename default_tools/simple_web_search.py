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
from cross_gpt import let_log, cacher, text_cutter

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

def is_reasonable_text(text, min_ratio=0.3):
    """
    Проверяет, выглядит ли текст разумным (не кракозябры).
    Возвращает True, если текст содержит достаточно "нормальных" символов.
    """
    if not text or len(text) < 10: return False
    total_chars = len(text)
    letters = sum(1 for c in text if c.isalpha())
    spaces = text.count(' ')
    punctuation = sum(1 for c in text if c in '.,!?;:-()[]{}"\'' and c.isprintable())
    replacement_chars = text.count('�')
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    good_chars = letters + spaces + punctuation
    good_ratio = good_chars / total_chars if total_chars > 0 else 0
    bad_chars = replacement_chars + control_chars
    bad_ratio = bad_chars / total_chars if total_chars > 0 else 1
    return (good_ratio >= min_ratio and 
            bad_ratio < 0.1 and 
            letters > 10 and 
            letters > total_chars * 0.1)

def decode_with_fallback(content):
    """Пробует разные стратегии декодирования сложного контента"""
    strategies = [
        lambda: content.decode('utf-8', errors='strict'),
        lambda: content.decode('utf-8-sig', errors='strict'),
        lambda: content.decode('windows-1251', errors='strict'),
        lambda: content.decode('cp1251', errors='strict'),
        lambda: content.decode('koi8-r', errors='strict'),
        lambda: content.decode('iso-8859-1', errors='strict'),
        lambda: content.decode('cp1252', errors='strict'),
    ]
    for strategy in strategies:
        try:
            result = strategy()
            if is_reasonable_text(result[:2000]): return result
        except: continue
    return content.decode('utf-8', errors='replace')

def decode_by_content_heuristic(content):
    """Эвристическое определение кодировки по содержимому"""
    cyrillic_bytes = sum(1 for b in content[:1000] if 0xC0 <= b <= 0xFF or 0x80 <= b <= 0xBF)
    if cyrillic_bytes > 100:
        for enc in ['windows-1251', 'cp1251', 'koi8-r', 'koi8-u']:
            try: return content.decode(enc, errors='strict')
            except: continue
    return content.decode('utf-8', errors='replace')

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

def fetch_links_ddg(query, max_results=10):
    results = []
    try:
        with DDGS() as ddgs: # Use generator to get results
            for r in ddgs.text(query, max_results=max_results):
                results.append(r['href'])
                if len(results) >= max_results: break
    except Exception as e: pass
    return results if results else []

def get_page_text(url):
    try:
        time.sleep(random.uniform(1, 3))
        session = requests_retry_session(retries=2, status_forcelist=(403, 500, 502, 504))
        headers = HEADERS.copy()
        headers['User-Agent'] = random.choice(USER_AGENTS)
        try:
            parsed_url = urlparse(url)
            headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}/"
        except Exception as e:
            let_log(f'[get_page_text] Error parsing URL {url}: {str(e)}')
            headers['Referer'] = 'https://www.google.com/'
        r = session.get(url, headers=headers, timeout=(10, 30), stream=True)
        raw_content = r.content
        encoding = None
        detection_method = "unknown"
        content_type = r.headers.get('Content-Type', '').lower()
        if 'charset=' in content_type:
            try:
                encoding = content_type.split('charset=')[-1].split(';')[0].strip().strip('"\'').lower()
                encoding = encoding.replace('_', '-').replace(' ', '-')
                detection_method = "HTTP header"
            except Exception as e: pass
        if not encoding:
            try:
                sample_size = min(len(raw_content), 10240)
                sample = raw_content[:sample_size].decode('ascii', errors='ignore')
                charset_patterns = [
                    r'<meta[^>]*charset=["\']?([^"\'>]+)["\']?',
                    r'<meta[^>]*content=["\'][^"\']*charset=([^"\';\s]+)',
                    r'xml:encoding=["\']?([^"\'>]+)["\']?'
                ]
                for pattern in charset_patterns:
                    matches = re.findall(pattern, sample, re.IGNORECASE)
                    if matches:
                        encoding = matches[0].strip().lower()
                        encoding = encoding.replace('_', '-').replace(' ', '-')
                        detection_method = "HTML meta tag"
                        break
            except Exception as e: pass
        if not encoding:
            try:
                encoding = r.apparent_encoding
                if encoding: detection_method = "requests.apparent_encoding"
            except Exception as e: pass
        if not encoding and len(raw_content) >= 4:
            bom_dict = {
                b'\xef\xbb\xbf': 'utf-8',
                b'\xff\xfe': 'utf-16-le',
                b'\xfe\xff': 'utf-16-be',
                b'\xff\xfe\x00\x00': 'utf-32-le',
                b'\x00\x00\xfe\xff': 'utf-32-be',
            }
            for bom, bom_encoding in bom_dict.items():
                if raw_content.startswith(bom):
                    encoding = bom_encoding
                    detection_method = "BOM detection"
                    break
        validated_encoding = None
        decoded_content = None
        common_encodings = [
            'utf-8', 'utf-8-sig',
            'windows-1251', 'cp1251',
            'koi8-r', 'koi8-u',
            'iso-8859-1', 'latin-1',
            'cp1252', 'windows-1252',
            'utf-16', 'utf-16-le', 'utf-16-be',
            'ascii'
        ]
        if encoding:
            encoding_lower = encoding.lower().replace('_', '-').replace(' ', '-')
            try:
                import codecs
                codecs.lookup(encoding_lower)
                try:
                    decoded_content = raw_content.decode(encoding_lower, errors='strict')
                    if is_reasonable_text(decoded_content[:1000]): validated_encoding = encoding_lower
                    else: decoded_content = None
                except UnicodeDecodeError:
                    decoded_content = raw_content.decode(encoding_lower, errors='replace')
                    if is_reasonable_text(decoded_content[:1000]): validated_encoding = encoding_lower
                    else: decoded_content = None
            except (LookupError, ValueError) as e: pass
        if not validated_encoding:
            for enc in common_encodings:
                try:
                    decoded_content = raw_content.decode(enc, errors='strict')
                    if is_reasonable_text(decoded_content[:1000]):
                        validated_encoding = enc
                        detection_method = f"common encoding fallback: {enc}"
                        break
                except UnicodeDecodeError:
                    try:
                        decoded_content = raw_content.decode(enc, errors='replace')
                        if is_reasonable_text(decoded_content[:1000]):
                            replacement_count = decoded_content[:1000].count('�')
                            if replacement_count < 50:
                                validated_encoding = enc
                                detection_method = f"common encoding fallback (replace): {enc}"
                                break
                    except Exception:
                        continue
        
        if not validated_encoding:
            import codecs
            all_encodings = [enc for enc in codecs.lookup_errors('strict')._codecs_dict.keys()]
            for enc in all_encodings:
                try:
                    if enc not in common_encodings:
                        decoded_content = raw_content.decode(enc, errors='strict')
                        if is_reasonable_text(decoded_content[:1000]):
                            validated_encoding = enc
                            detection_method = f"brute force: {enc}"
                            break
                except: continue
        if not validated_encoding:
            validated_encoding = 'utf-8'
            detection_method = "default utf-8 with replace"
            decoded_content = raw_content.decode(validated_encoding, errors='replace')
        if decoded_content:
            replacement_chars = decoded_content.count('�')
            total_chars = len(decoded_content)
            if total_chars > 0:
                replacement_ratio = replacement_chars / total_chars
                if replacement_ratio > 0.3: return ''
        soup = BeautifulSoup(decoded_content, 'html.parser')
        # Special handling for 403
        if r.status_code == 403:
            headers['User-Agent'] = random.choice(USER_AGENTS)
            r = session.get(url, headers=headers, timeout=(10, 30))
        if not r.ok: return ''
        content_type = r.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type: return ''
        # Пункт 3: Установите правильную кодировку
        try: r.encoding = validated_encoding if validated_encoding else 'utf-8'
        except Exception as e: r.encoding = 'utf-8'
        for script in soup(["script", "style", "noscript"]): script.decompose()
        for element in soup(["nav", "footer", "header", "aside", "form"]): element.decompose()
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
                if found and len(found.get_text(strip=True)) > 100:
                    main_content = found
                    break
            except Exception as e: continue
        if main_content: text = main_content.get_text(" ", strip=True)
        else: 
            all_elements = soup.find_all(['div', 'section'])
            best_element = None
            max_text_length = 0
            for element in all_elements:
                element_text = element.get_text(strip=True)
                if len(element_text) > max_text_length and len(element_text) > 200:
                    max_text_length = len(element_text)
                    best_element = element
            if best_element: text = best_element.get_text(" ", strip=True)
            else: text = soup.get_text(" ", strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 100: return ''
        words = text.split()
        if len(words) < 20: return ''
        if any(bad_text in text.lower() for bad_text in ['404', 'not found', 'page not found', 'access denied']): return ''
        return text
    except Exception as e: return ''

@cacher
def pages_handler(text):
    # Константы
    TARGET_SUCCESSFUL_SITES = 10
    MAX_ADDITIONAL_REQUESTS = 5  # Максимальное количество дополнительных запросов к поисковику
    MAX_TOTAL_LINKS = 30  # Максимальное общее количество ссылок для обработки
    all_links = []
    successful_texts = []  # Список успешных текстов [(url, text), ...]
    processed_links = set()  # Множество обработанных ссылок
    additional_requests_made = 0  # Счетчик дополнительных запросов
    # Функция для получения дополнительных ссылок
    def get_more_links(count):
        nonlocal additional_requests_made
        if additional_requests_made >= MAX_ADDITIONAL_REQUESTS: return []
        try:
            additional_requests_made += 1
            # Запрашиваем на count больше, так как некоторые могут не сработать
            new_links = fetch_links_ddg(text, max_results=count + 2)
            # Исключаем уже обработанные ссылки
            fresh_links = [link for link in new_links if link not in processed_links]
            return fresh_links
        except: return []
    # Получаем первоначальные ссылки
    try:
        initial_links = fetch_links_ddg(text, LINKS_PER_ENGINE)
        all_links.extend(initial_links)
    except Exception as e: let_log(f'{main.search_error_msg}{str(e)}'); return f'{main.search_error_msg}{str(e)}'
    if not all_links: return main.no_results_msg
    # Основной цикл обработки ссылок
    link_index = 0
    while len(successful_texts) < TARGET_SUCCESSFUL_SITES and link_index < len(all_links):
        # Проверяем, не превысили ли максимальное количество ссылок
        if len(processed_links) >= MAX_TOTAL_LINKS:
            let_log(f'[main] Достигнут максимальный лимит ссылок ({MAX_TOTAL_LINKS})')
            break
        link = all_links[link_index]
        # Пропускаем уже обработанные ссылки (на всякий случай)
        if link in processed_links:
            link_index += 1
            continue
        page_text = get_page_text(link)
        processed_links.add(link)
        if page_text: successful_texts.append((link, page_text))
        else:
            # Если текст не получен, запрашиваем дополнительную ссылку
            if len(successful_texts) < TARGET_SUCCESSFUL_SITES:
                additional_links = get_more_links(1)
                if additional_links: all_links.extend(additional_links)
        link_index += 1
        # Пауза между запросами (кроме последнего)
        if link_index < len(all_links) and link_index < MAX_TOTAL_LINKS:
            sleep_time = random.uniform(2, 4)
            time.sleep(sleep_time)
    # Формируем итоговый текст
    if not successful_texts: return main.search_failed
    collected_text = ""
    for link, text in successful_texts[:TARGET_SUCCESSFUL_SITES]: collected_text += f"=== {link} ===\n{text}\n\n"
    let_log(f'Обработано ссылок: {len(processed_links)}, всего получено ссылок: {len(all_links)}')
    return collected_text

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
    return main.results_prefix + text_cutter(pages_handler(text))