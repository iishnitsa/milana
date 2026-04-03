'''
search_web
send 1 search query after the command; returns summaries of several websites
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
from cross_gpt import load_info_loaders, default_handlers_names, found_info_1

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
    if not text or len(text) < 10: 
        let_log(f'[is_reasonable_text] Text too short: {len(text) if text else 0} chars')
        return False
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
    result = (good_ratio >= min_ratio and 
            bad_ratio < 0.1 and 
            letters > 10 and 
            letters > total_chars * 0.1)
    let_log(f'[is_reasonable_text] good_ratio={good_ratio:.2f}, bad_ratio={bad_ratio:.2f}, letters={letters}, result={result}')
    return result

def decode_with_fallback(content):
    """Пробует разные стратегии декодирования сложного контента"""
    let_log(f'[decode_with_fallback] Trying to decode content of length {len(content)}')
    strategies = [
        lambda: content.decode('utf-8', errors='strict'),
        lambda: content.decode('utf-8-sig', errors='strict'),
        lambda: content.decode('windows-1251', errors='strict'),
        lambda: content.decode('cp1251', errors='strict'),
        lambda: content.decode('koi8-r', errors='strict'),
        lambda: content.decode('iso-8859-1', errors='strict'),
        lambda: content.decode('cp1252', errors='strict'),
    ]
    for i, strategy in enumerate(strategies):
        try:
            result = strategy()
            if is_reasonable_text(result[:2000]): 
                let_log(f'[decode_with_fallback] Success with strategy {i}')
                return result
        except Exception as e:
            let_log(f'[decode_with_fallback] Strategy {i} failed: {str(e)}')
            continue
    let_log('[decode_with_fallback] All strategies failed, using utf-8 with replace')
    return content.decode('utf-8', errors='replace')

def decode_by_content_heuristic(content):
    """Эвристическое определение кодировки по содержимому"""
    let_log(f'[decode_by_content_heuristic] Analyzing content of length {len(content)}')
    cyrillic_bytes = sum(1 for b in content[:1000] if 0xC0 <= b <= 0xFF or 0x80 <= b <= 0xBF)
    let_log(f'[decode_by_content_heuristic] Cyrillic bytes count: {cyrillic_bytes}')
    if cyrillic_bytes > 100:
        let_log('[decode_by_content_heuristic] High cyrillic content detected, trying windows-1251')
        for enc in ['windows-1251', 'cp1251', 'koi8-r', 'koi8-u']:
            try: 
                result = content.decode(enc, errors='strict')
                let_log(f'[decode_by_content_heuristic] Success with {enc}')
                return result
            except Exception as e:
                let_log(f'[decode_by_content_heuristic] {enc} failed: {str(e)}')
                continue
    let_log('[decode_by_content_heuristic] Using utf-8 with replace')
    return content.decode('utf-8', errors='replace')

def requests_retry_session(
    retries=3,
    backoff_factor=0.5,
    status_forcelist=(500, 502, 504, 403),
    session=None,
):
    """Creates a requests session with retry logic"""
    let_log(f'[requests_retry_session] Creating session with retries={retries}, backoff={backoff_factor}')
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
    let_log('[requests_retry_session] Session created successfully')
    return session

def fetch_links_ddg(query, max_results=10):
    let_log(f'[fetch_links_ddg] Searching for: "{query[:50]}...", max_results={max_results}')
    results = []
    try:
        # Добавлен таймаут 60 секунд, явно указан backend="auto"
        with DDGS(timeout=60) as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results, backend="auto")):
                results.append(r['href'])
                let_log(f'[fetch_links_ddg] Found result {i+1}: {r["href"][:100]}...')
                if len(results) >= max_results: break
        let_log(f'[fetch_links_ddg] Total links found: {len(results)}')
    except Exception as e: 
        let_log(f'[fetch_links_ddg] Error: {str(e)}')
        pass
    return results if results else []

def get_page_text(url):
    let_log(f'[get_page_text] Processing URL: {url}')
    try:
        time.sleep(random.uniform(1, 3))
        let_log(f'[get_page_text] Sleep completed')
        session = requests_retry_session(retries=2, status_forcelist=(403, 500, 502, 504))
        headers = HEADERS.copy()
        user_agent = random.choice(USER_AGENTS)
        headers['User-Agent'] = user_agent
        let_log(f'[get_page_text] Using User-Agent: {user_agent[:50]}...')
        try:
            parsed_url = urlparse(url)
            headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}/"
            let_log(f'[get_page_text] Set Referer: {headers["Referer"]}')
        except Exception as e:
            let_log(f'[get_page_text] Error parsing URL {url}: {str(e)}')
            headers['Referer'] = 'https://www.google.com/'
        let_log(f'[get_page_text] Making request to {url}')
        r = session.get(url, headers=headers, timeout=(10, 30), stream=True)
        let_log(f'[get_page_text] Response status: {r.status_code}')
        
        # --- Обработка PDF файлов ---
        content_type = r.headers.get('Content-Type', '').lower()
        let_log(f'[get_page_text] Content-Type: {content_type}')
        if 'application/pdf' in content_type:
            let_log('[get_page_text] PDF DETECTED on site!')
            from cross_gpt import input_info_loaders
            # Проверяем, загружен ли обработчик PDF
            if 'pdf' not in input_info_loaders:
                let_log('[get_page_text] PDF handler not loaded, loading now')
                # Загружаем обработчики файлов один раз
                load_info_loaders(default_handlers_names)
            pdf_handler = input_info_loaders.get('pdf')
            if pdf_handler:
                try:
                    let_log(f'[get_page_text] Processing PDF content, size: {len(r.content)} bytes')
                    text_result = pdf_handler(r.content, input_info_loaders)
                    if text_result and len(text_result) > 100 and is_reasonable_text(text_result[:2000]):
                        let_log(f'[get_page_text] PDF processed successfully, text length: {len(text_result)}')
                        return text_result
                    else:
                        let_log(f"[get_page_text] PDF processing returned insufficient text for {url}")
                        return ''
                except Exception as e:
                    let_log(f"[get_page_text] Error processing PDF {url}: {e}")
                    return ''
            else:
                let_log("[get_page_text] PDF handler not available, skipping PDF")
                return ''
        # --- Конец обработки PDF ---

        raw_content = r.content
        let_log(f'[get_page_text] Raw content size: {len(raw_content)} bytes')
        encoding = None
        detection_method = "unknown"
        content_type = r.headers.get('Content-Type', '').lower()
        if 'charset=' in content_type:
            try:
                encoding = content_type.split('charset=')[-1].split(';')[0].strip().strip('"\'').lower()
                encoding = encoding.replace('_', '-').replace(' ', '-')
                detection_method = "HTTP header"
                let_log(f'[get_page_text] Found charset in HTTP header: {encoding}')
            except Exception as e:
                let_log(f'[get_page_text] Error parsing charset from header: {e}')
                pass
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
                        let_log(f'[get_page_text] Found charset in HTML meta tag: {encoding}')
                        break
            except Exception as e:
                let_log(f'[get_page_text] Error detecting charset from HTML: {e}')
                pass
        if not encoding:
            try:
                encoding = r.apparent_encoding
                if encoding: 
                    detection_method = "requests.apparent_encoding"
                    let_log(f'[get_page_text] Using apparent_encoding: {encoding}')
            except Exception as e:
                let_log(f'[get_page_text] Error getting apparent_encoding: {e}')
                pass
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
                    let_log(f'[get_page_text] Detected BOM for encoding: {encoding}')
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
                    if is_reasonable_text(decoded_content[:1000]): 
                        validated_encoding = encoding_lower
                        let_log(f'[get_page_text] Successfully validated encoding: {validated_encoding} (strict)')
                    else: 
                        decoded_content = None
                        let_log(f'[get_page_text] Encoding {encoding_lower} produced unreasonable text (strict)')
                except UnicodeDecodeError:
                    let_log(f'[get_page_text] Strict decode failed for {encoding_lower}, trying with replace')
                    decoded_content = raw_content.decode(encoding_lower, errors='replace')
                    if is_reasonable_text(decoded_content[:1000]): 
                        validated_encoding = encoding_lower
                        let_log(f'[get_page_text] Successfully validated encoding: {validated_encoding} (replace)')
                    else: 
                        decoded_content = None
                        let_log(f'[get_page_text] Encoding {encoding_lower} produced unreasonable text (replace)')
            except (LookupError, ValueError) as e:
                let_log(f'[get_page_text] Invalid encoding {encoding_lower}: {e}')
                pass
        if not validated_encoding:
            let_log('[get_page_text] Trying common encodings')
            for enc in common_encodings:
                try:
                    decoded_content = raw_content.decode(enc, errors='strict')
                    if is_reasonable_text(decoded_content[:1000]):
                        validated_encoding = enc
                        detection_method = f"common encoding fallback: {enc}"
                        let_log(f'[get_page_text] Found valid encoding: {enc} (strict)')
                        break
                except UnicodeDecodeError:
                    try:
                        decoded_content = raw_content.decode(enc, errors='replace')
                        if is_reasonable_text(decoded_content[:1000]):
                            replacement_count = decoded_content[:1000].count('�')
                            if replacement_count < 50:
                                validated_encoding = enc
                                detection_method = f"common encoding fallback (replace): {enc}"
                                let_log(f'[get_page_text] Found valid encoding: {enc} (replace, replacements={replacement_count})')
                                break
                    except Exception as e:
                        let_log(f'[get_page_text] Error with encoding {enc}: {e}')
                        continue
        
        if not validated_encoding:
            let_log('[get_page_text] Trying brute force encoding detection')
            import codecs
            all_encodings = [enc for enc in codecs.lookup_errors('strict')._codecs_dict.keys()]
            for enc in all_encodings:
                try:
                    if enc not in common_encodings:
                        decoded_content = raw_content.decode(enc, errors='strict')
                        if is_reasonable_text(decoded_content[:1000]):
                            validated_encoding = enc
                            detection_method = f"brute force: {enc}"
                            let_log(f'[get_page_text] Found valid encoding via brute force: {enc}')
                            break
                except:
                    continue
        if not validated_encoding:
            validated_encoding = 'utf-8'
            detection_method = "default utf-8 with replace"
            decoded_content = raw_content.decode(validated_encoding, errors='replace')
            let_log(f'[get_page_text] Using default utf-8 with replace')
        if decoded_content:
            replacement_chars = decoded_content.count('�')
            total_chars = len(decoded_content)
            if total_chars > 0:
                replacement_ratio = replacement_chars / total_chars
                let_log(f'[get_page_text] Replacement ratio: {replacement_ratio:.2f} ({replacement_chars}/{total_chars})')
                if replacement_ratio > 0.3:
                    let_log('[get_page_text] Too many replacement characters, returning empty')
                    return ''
        soup = BeautifulSoup(decoded_content, 'html.parser')
        # Special handling for 403
        if r.status_code == 403:
            let_log('[get_page_text] Got 403, retrying with different User-Agent')
            headers['User-Agent'] = random.choice(USER_AGENTS)
            r = session.get(url, headers=headers, timeout=(10, 30))
            let_log(f'[get_page_text] Retry status: {r.status_code}')
        if not r.ok: 
            let_log(f'[get_page_text] Response not OK: {r.status_code}')
            return ''
        content_type = r.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            let_log(f'[get_page_text] Not HTML content: {content_type}')
            return ''
        # Пункт 3: Установите правильную кодировку
        try: 
            r.encoding = validated_encoding if validated_encoding else 'utf-8'
            let_log(f'[get_page_text] Set response encoding to: {r.encoding}')
        except Exception as e: 
            r.encoding = 'utf-8'
            let_log(f'[get_page_text] Error setting encoding, using utf-8: {e}')
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
                    let_log(f'[get_page_text] Found main content with selector: {selector}')
                    break
            except Exception as e:
                let_log(f'[get_page_text] Error with selector {selector}: {e}')
                continue
        if main_content: 
            text = main_content.get_text(" ", strip=True)
            let_log(f'[get_page_text] Extracted text from main content, length: {len(text)}')
        else:
            let_log('[get_page_text] No main content found, trying alternative method')
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
                let_log(f'[get_page_text] Found best element with text length: {max_text_length}')
            else: 
                text = soup.get_text(" ", strip=True)
                let_log(f'[get_page_text] Using full soup text, length: {len(text)}')
        text = re.sub(r'\s+', ' ', text).strip()
        let_log(f'[get_page_text] Final text length after cleaning: {len(text)}')
        if len(text) < 100: 
            let_log(f'[get_page_text] Text too short ({len(text)} chars), returning empty')
            return ''
        words = text.split()
        if len(words) < 20: 
            let_log(f'[get_page_text] Too few words ({len(words)}), returning empty')
            return ''
        bad_phrases = ['404', 'not found', 'page not found', 'access denied']
        if any(bad_text in text.lower() for bad_text in bad_phrases):
            let_log('[get_page_text] Found error message in text, returning empty')
            return ''
        let_log(f'[get_page_text] Successfully extracted text, length: {len(text)}, words: {len(words)}')
        return text
    except Exception as e: 
        let_log(f'[get_page_text] Error processing {url}: {str(e)}')
        return ''

def main(text):
    if not hasattr(main, 'attr_names'):
        let_log('INITIALIZATION')
        main.attr_names = (
            'search_error_msg',
            'used_links_header',
        )
        main.search_error_msg = 'Search error: '
        main.used_links_header = 'Used links:'
        return
    let_log('WEB SEARCH CALLED')
    let_log(f'Search query: {text}')
    @cacher
    def pages_handler(text):
        let_log(f'[pages_handler] Starting search for: {text[:100]}...')
        # Добавляем общий try-except, чтобы при любой ошибке возвращать пустой список
        try:
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
                if additional_requests_made >= MAX_ADDITIONAL_REQUESTS:
                    let_log(f'[get_more_links] Reached max additional requests ({MAX_ADDITIONAL_REQUESTS})')
                    return []
                try:
                    additional_requests_made += 1
                    let_log(f'[get_more_links] Making additional request #{additional_requests_made} for {count} links')
                    # Запрашиваем на count больше, так как некоторые могут не сработать
                    new_links = fetch_links_ddg(text, max_results=count + 2)
                    # Исключаем уже обработанные ссылки
                    fresh_links = [link for link in new_links if link not in processed_links]
                    let_log(f'[get_more_links] Got {len(fresh_links)} fresh links out of {len(new_links)}')
                    return fresh_links
                except Exception as e:
                    let_log(f'[get_more_links] Error: {e}')
                    return []
            # Получаем первоначальные ссылки
            try:
                let_log('[main] Fetching initial links')
                initial_links = fetch_links_ddg(text, LINKS_PER_ENGINE)
                all_links.extend(initial_links)
                let_log(f'[main] Initial links count: {len(initial_links)}')
            except Exception as e: 
                let_log(f'{main.search_error_msg}{str(e)}')
                return []   # возвращаем пустой список при ошибке
            if not all_links:
                let_log('[main] No links found')
                return []   # возвращаем пустой список, если нет ссылок
            # Основной цикл обработки ссылок
            link_index = 0
            let_log(f'[main] Starting main loop, target successful sites: {TARGET_SUCCESSFUL_SITES}')
            while len(successful_texts) < TARGET_SUCCESSFUL_SITES and link_index < len(all_links):
                # Проверяем, не превысили ли максимальное количество ссылок
                if len(processed_links) >= MAX_TOTAL_LINKS:
                    let_log(f'[main] Reached max total links limit ({MAX_TOTAL_LINKS})')
                    break
                link = all_links[link_index]
                let_log(f'[main] Processing link #{link_index + 1}/{len(all_links)}: {link[:100]}...')
                # Пропускаем уже обработанные ссылки (на всякий случай)
                if link in processed_links:
                    let_log(f'[main] Link already processed, skipping')
                    link_index += 1
                    continue
                page_text = get_page_text(link)
                processed_links.add(link)
                if page_text: 
                    successful_texts.append((link, page_text))
                    let_log(f'[main] Success! Got text from {link[:100]}... (total successful: {len(successful_texts)})')
                else:
                    let_log(f'[main] No text from {link[:100]}...')
                    # Если текст не получен, запрашиваем дополнительную ссылку
                    if len(successful_texts) < TARGET_SUCCESSFUL_SITES:
                        additional_links = get_more_links(1)
                        if additional_links:
                            all_links.extend(additional_links)
                            let_log(f'[main] Added {len(additional_links)} more links, total links now: {len(all_links)}')
                link_index += 1
                # Пауза между запросами (кроме последнего)
                if link_index < len(all_links) and link_index < MAX_TOTAL_LINKS:
                    sleep_time = random.uniform(2, 4)
                    let_log(f'[main] Sleeping for {sleep_time:.1f}s before next request')
                    time.sleep(sleep_time)
            # Возвращаем список успешных текстов (без сокращения)
            if not successful_texts:
                let_log('[main] No successful texts found')
                return []   # пустой список, если ничего не удалось получить
            let_log(f'[main] Processed links: {len(processed_links)}, total links received: {len(all_links)}')
            return successful_texts[:TARGET_SUCCESSFUL_SITES]
        except Exception as e:
            let_log(f'[pages_handler] Unexpected error: {str(e)}')
            return []  # при любой ошибке возвращаем пустой список

    # Получаем список (url, raw_text) из кэшируемой функции
    sites_data = pages_handler(text)
    if not sites_data:
        return found_info_1

    # Объединяем все сырые тексты в одну строку
    combined_raw = "\n\n".join(raw for _, raw in sites_data)
    # Сжимаем объединённый текст (text_cutter вызывается вне pages_handler)
    summarized = text_cutter(text_cutter(combined_raw))

    # Формируем список ссылок
    links = [url for url, _ in sites_data]
    # Добавляем заголовок из атрибута (с возможной локализацией)
    header = main.used_links_header
    links_str = "\n".join(links)
    result = f"{summarized}\n\n{header}\n{links_str}"
    let_log(f'[main] Final result length: {len(result)} chars')
    return result