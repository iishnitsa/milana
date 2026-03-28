'''
web_page_text
send 1 URL; returns the text from the web page
Get web page by URL
Extracts main text content from a web page
'''

import re
import requests
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from cross_gpt import let_log, found_info_1

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0'
]

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

def is_reasonable_text(text, min_ratio=0.3):
    if not text or len(text) < 10:
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
    return (good_ratio >= min_ratio and
            bad_ratio < 0.1 and
            letters > 10 and
            letters > total_chars * 0.1)

def requests_retry_session(retries=3, backoff_factor=0.5,
                           status_forcelist=(500, 502, 504, 403),
                           session=None):
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

def get_page_text(url):
    try:
        time.sleep(random.uniform(1, 3))
        session = requests_retry_session(retries=2, status_forcelist=(403, 500, 502, 504))
        headers = HEADERS.copy()
        user_agent = random.choice(USER_AGENTS)
        headers['User-Agent'] = user_agent

        try:
            parsed_url = urlparse(url)
            headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}/"
        except Exception:
            headers['Referer'] = 'https://www.google.com/'

        r = session.get(url, headers=headers, timeout=(10, 30), stream=True)
        content_type = r.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            return ''

        raw_content = r.content
        encoding = None
        if 'charset=' in content_type:
            try:
                encoding = content_type.split('charset=')[-1].split(';')[0].strip().strip('"\'').lower()
                encoding = encoding.replace('_', '-').replace(' ', '-')
            except Exception:
                pass

        if not encoding:
            try:
                sample = raw_content[:min(len(raw_content), 10240)].decode('ascii', errors='ignore')
                meta_patterns = [
                    r'<meta[^>]*charset=["\']?([^"\'>]+)["\']?',
                    r'<meta[^>]*content=["\'][^"\']*charset=([^"\';\s]+)',
                ]
                for pattern in meta_patterns:
                    match = re.search(pattern, sample, re.IGNORECASE)
                    if match:
                        encoding = match.group(1).strip().lower().replace('_', '-').replace(' ', '-')
                        break
            except Exception:
                pass

        if not encoding:
            encoding = r.apparent_encoding

        if not encoding:
            encoding = 'utf-8'

        try:
            text = raw_content.decode(encoding, errors='strict')
            if not is_reasonable_text(text[:2000]):
                text = raw_content.decode(encoding, errors='replace')
        except UnicodeDecodeError:
            text = raw_content.decode(encoding, errors='replace')

        soup = BeautifulSoup(text, 'html.parser')
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        main_selectors = [
            'main', 'article', '[role="main"]', '.main-content', '.article-content',
            '.post-content', '.entry-content', '#content', '.content', '#main', '.main',
            '.story', '.story-content', '#article', '.article'
        ]
        main_content = None
        for selector in main_selectors:
            try:
                elem = soup.select_one(selector)
                if elem and len(elem.get_text(strip=True)) > 100:
                    main_content = elem
                    break
            except Exception:
                continue

        if main_content:
            text_content = main_content.get_text(" ", strip=True)
        else:
            candidates = soup.find_all(['div', 'section'])
            best_elem = None
            best_len = 0
            for elem in candidates:
                elem_text = elem.get_text(strip=True)
                if len(elem_text) > best_len and len(elem_text) > 200:
                    best_len = len(elem_text)
                    best_elem = elem
            if best_elem:
                text_content = best_elem.get_text(" ", strip=True)
            else:
                text_content = soup.get_text(" ", strip=True)

        text_content = re.sub(r'\s+', ' ', text_content).strip()

        if len(text_content) < 100:
            return ''
        words = text_content.split()
        if len(words) < 20:
            return ''
        error_phrases = ['404', 'not found', 'page not found', 'access denied']
        if any(phrase in text_content.lower() for phrase in error_phrases):
            return ''

        return text_content
    except Exception as e:
        return ''

def main(text):
    if not hasattr(main, 'attr_names'):
        let_log('INITIALIZATION')
        main.attr_names = (
            'extract_error_msg',
        )
        main.extract_error_msg = 'Extraction error: '
        return
    let_log('WEB PAGE TEXT CALLED')
    let_log(f'URL: {text}')
    url = text.strip()
    try:
        page_text = get_page_text(url)
        if not page_text:
            return found_info_1
        return page_text
    except Exception as e:
        let_log(f'Error in web_page_text: {str(e)}')
        return f"{main.extract_error_msg}{str(e)}"