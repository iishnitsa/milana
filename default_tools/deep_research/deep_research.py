'''
deep_research
conducts iterative deep research: generates search queries, gathers information, analyzes, reflects on gaps, and continues until sufficient depth is reached. Use for complex questions requiring deep synthesis.
Deep Research
Performs iterative multi-query web research with reflection and gap analysis. Recommended for complex topics needing deep understanding.
'''

import re
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from ddgs import DDGS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from cross_gpt import let_log, cacher, text_cutter, ask_model, found_info_1

# --- Constants (not localized, not in main.attr_names) ---
MAX_ITERATIONS = 3           # maximum number of research cycles
QUERIES_PER_ITERATION = 3    # how many search queries to generate per cycle
MAX_PAGES_PER_QUERY = 2      # how many links to fetch per query
MAX_TOTAL_PAGES = 12         # total pages to fetch across all cycles (safety limit)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',]

def requests_retry_session(retries=2, backoff_factor=0.5, status_forcelist=(500, 502, 504, 403), session=None):
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist, allowed_methods={"GET", "POST"})
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def is_reasonable_text(text, min_ratio=0.3):
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
    return (good_ratio >= min_ratio and bad_ratio < 0.1 and letters > 10 and letters > total_chars * 0.1)

def fetch_links_ddg(query, max_results=5):
    let_log(f'[fetch_links_ddg] Searching: {query[:50]}...')
    results = []
    try:
        with DDGS(timeout=60) as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results, backend="auto")):
                results.append(r['href'])
                if len(results) >= max_results: break
    except Exception as e: let_log(f'[fetch_links_ddg] Error: {e}')
    return results

def get_page_text(url):
    let_log(f'[get_page_text] Fetching {url[:80]}...')
    try:
        time.sleep(random.uniform(1, 2))
        session = requests_retry_session(retries=1)
        headers = HEADERS.copy()
        headers['User-Agent'] = random.choice(USER_AGENTS)
        parsed = urlparse(url)
        headers['Referer'] = f"{parsed.scheme}://{parsed.netloc}/"
        r = session.get(url, headers=headers, timeout=(10, 20))
        if r.status_code != 200: return ''
        content_type = r.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type: return ''
        encoding = r.encoding or 'utf-8'
        if 'charset=' in content_type:
            try:
                enc = content_type.split('charset=')[-1].split(';')[0].strip()
                if enc: encoding = enc
            except: pass
        try: html = r.content.decode(encoding, errors='replace')
        except: html = r.text
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]): tag.decompose()
        main_candidates = soup.find_all(['main', 'article', 'div'], class_=re.compile(r'content|main|article|post'))
        best_text = ''
        for candidate in main_candidates:
            text = candidate.get_text(' ', strip=True)
            if len(text) > len(best_text): best_text = text
        if not best_text or len(best_text) < 100: best_text = soup.get_text(' ', strip=True)
        best_text = re.sub(r'\s+', ' ', best_text).strip()
        if len(best_text) < 100: return ''
        return best_text[:8000]
    except Exception as e: let_log(f'[get_page_text] Error: {e}'); return ''

# --- Cachable data fetching (no ask_model or text_cutter inside) ---

@cacher
def fetch_data_for_queries(queries, max_pages_total=MAX_TOTAL_PAGES):
    """
    Fetches web pages for given queries. Returns a list of (url, text).
    This function is cached because it only does HTTP requests and parsing.
    """
    let_log(f'[fetch_data_for_queries] Starting data fetch for {len(queries)} queries')
    all_data = []  # list of (url, text)
    pages_fetched = 0
    for q in queries:
        if pages_fetched >= max_pages_total: break
        links = fetch_links_ddg(q, max_results=MAX_PAGES_PER_QUERY)
        let_log(f'[fetch_data_for_queries] Query "{q}" returned {len(links)} links')
        for link in links:
            if pages_fetched >= max_pages_total: break
            text = get_page_text(link)
            if text and len(text) > 200:
                all_data.append((link, text))
                pages_fetched += 1
                let_log(f'[fetch_data_for_queries] Fetched page {pages_fetched}: {link[:60]}... length {len(text)}')
            time.sleep(random.uniform(0.5, 1.5))
    let_log(f'[fetch_data_for_queries] Total pages fetched: {pages_fetched}')
    return all_data

def generate_initial_queries(topic, prompt):
    """Generate initial search queries using LLM."""
    let_log('[generate_initial_queries] Generating queries...')
    try:
        queries_raw = ask_model(topic, system_prompt=prompt)
        queries = [q.strip() for q in queries_raw.split('\n') if q.strip() and len(q.strip()) > 3]
        if not queries: queries = [topic]
        queries = queries[:QUERIES_PER_ITERATION]
        let_log(f'[generate_initial_queries] Generated: {queries}')
        return queries
    except Exception as e: let_log(f'[generate_initial_queries] Failed: {e}'); raise

def generate_next_queries(topic, current_info_summary, gap_prompt, queries_prompt):
    """
    Analyze what's missing and generate next search queries.
    Returns a list of queries (or empty list if no more needed).
    """
    let_log('[generate_next_queries] Analyzing gaps and generating next queries...')
    # First, identify gaps
    gap_user_message = f"Original topic: {topic}\n\nInformation gathered so far:\n{current_info_summary}"
    try:
        gaps_raw = ask_model(gap_user_message, system_prompt=gap_prompt)
        gaps = [g.strip() for g in gaps_raw.split('\n') if g.strip()]
        if not gaps: return []
        let_log(f'[generate_next_queries] Identified gaps: {gaps}')
    except Exception as e: let_log(f'[generate_next_queries] Gap analysis failed: {e}'); return []
    # Then generate queries based on gaps
    queries_user_message = f"Original topic: {topic}\n\nKnowledge gaps identified:\n" + "\n".join(gaps)
    try:
        queries_raw = ask_model(queries_user_message, system_prompt=queries_prompt)
        queries = [q.strip() for q in queries_raw.split('\n') if q.strip() and len(q.strip()) > 3]
        queries = queries[:QUERIES_PER_ITERATION]
        let_log(f'[generate_next_queries] Generated follow-up queries: {queries}')
        return queries
    except Exception as e: let_log(f'[generate_next_queries] Query generation failed: {e}'); return []

def synthesize_answer(topic, all_data, synthesis_prompt):
    """Synthesize final answer from all gathered data."""
    let_log('[synthesize_answer] Synthesizing answer...')
    combined_parts = []
    for url, text in all_data: combined_parts.append(f"Source: {url}\n{text}")
    combined = "\n\n---\n\n".join(combined_parts)
    combined = text_cutter(combined)
    user_message = f"Original topic: {topic}\n\nInformation gathered from web pages:\n{combined}"
    try:
        answer = ask_model(user_message, system_prompt=synthesis_prompt)
        answer = re.sub(r'^[\s\n]+', '', answer)
        answer = re.sub(r'[\s\n]+$', '', answer)
        return answer
    except Exception as e: let_log(f'[synthesize_answer] Failed: {e}'); raise

def summarize_info(data):
    """Create a concise summary of gathered information for gap analysis."""
    if not data: return "No information gathered yet." # Just concatenate first 2000 chars of each text to keep it manageable
    parts = []
    for url, text in data: parts.append(f"From {url}:\n{text[:1500]}")
    combined = "\n\n".join(parts) # Use text_cutter to compress further (but this is outside cacher)
    return text_cutter(combined)

def main(text):
    if not hasattr(main, 'attr_names'):
        let_log('INITIALIZATION deep_research')
        main.attr_names = (
            'generate_queries_prompt',
            'gap_analysis_prompt',
            'followup_queries_prompt',
            'synthesis_prompt',
            'error_msg',
            'no_results_msg',)
        main.generate_queries_prompt = "You are an expert research assistant. Generate 3-5 specific search queries (one per line) that would help gather comprehensive information on the user's topic. Output only the queries, each on a new line. No numbering, no extra text."
        main.gap_analysis_prompt = "You are a research analyst. Analyze the information gathered so far and identify specific knowledge gaps that still need to be filled to provide a complete, well-researched answer to the original topic. List each gap on a new line as a clear, actionable question. If no significant gaps remain, output exactly the word 'NONE'."
        main.followup_queries_prompt = "You are a search query generator. Based on the identified knowledge gaps, generate 3-5 specific search queries (one per line) that would help fill those gaps. Output only the queries, each on a new line. No numbering, no extra text."
        main.synthesis_prompt = "You are a research assistant. Based on all the information provided by the user, produce a comprehensive, well-structured answer to the original question/topic. Include key facts, different perspectives if any, and cite sources by their URLs. Write in clear paragraphs, do not use markdown, be concise but thorough."
        main.error_msg = "An error occurred during the research."
        main.no_results_msg = "No relevant information found after multiple attempts."
        return
    let_log(f'[DeepResearch] main called with: {text[:200]}')
    topic = text.strip()
    try:
        # Step 1: Generate initial queries
        queries = generate_initial_queries(topic, main.generate_queries_prompt)
        # Step 2: Iterative research loop
        all_data = []  # list of (url, text)
        iteration = 0
        total_pages_fetched = 0
        while iteration < MAX_ITERATIONS:
            let_log(f'[DeepResearch] Iteration {iteration+1}/{MAX_ITERATIONS}')
            # Fetch data for current queries (cached)
            new_data = fetch_data_for_queries(queries)
            all_data.extend(new_data)
            total_pages_fetched = len(all_data)
            # If no new data, break
            if not new_data: let_log('[DeepResearch] No new data fetched, stopping.'); break
            # If we have enough data (some threshold), we can optionally break early
            if total_pages_fetched >= MAX_TOTAL_PAGES: let_log(f'[DeepResearch] Reached max total pages ({MAX_TOTAL_PAGES}), stopping.'); break
            # Summarize current information for gap analysis
            summary = summarize_info(all_data)
            # Generate next queries based on gaps
            next_queries = generate_next_queries( topic, summary, main.gap_analysis_prompt, main.followup_queries_prompt)
            if not next_queries: let_log('[DeepResearch] No more queries to generate, stopping iteration.'); break
            queries = next_queries
            iteration += 1
        if not all_data: let_log('[DeepResearch] No data collected at all'); return main.no_results_msg
        # Step 3: Synthesize final answer
        answer = synthesize_answer(topic, all_data, main.synthesis_prompt)
        return answer
    except Exception as e: let_log(f'[DeepResearch] Unexpected error: {e}'); return main.error_msg