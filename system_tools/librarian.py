# librarian.py
'''
нужна_информация
вызови для получения недостающей информации
'''

import os
import re
from cross_gpt import (
    global_state,
    ask_model,
    text_cutter,
    get_embs,
    coll_exec,
    let_log,
    found_info_1,
    parse_prompt_response,
    language,
    load_locale,
)

def _extract_first_digit(text, default):
    """Извлекает первую цифру из текста (аналог parse_prompt_response для готового ответа)."""
    if not text:
        return default
    for ch in text[:5]:
        if ch.isdigit():
            try:
                return int(ch)
            except ValueError:
                continue
    return default

def main(quest):
    # Инициализация атрибутов модуля при первом вызове
    if not hasattr(main, 'attr_names'):
        let_log('ЛИБРАРИАН')
        main.attr_names = (
            'best_result_l_1',
            'best_result_l_2',
            'request_l_1',
            'information_l_1',
            'satisfies_l_1',
            'select_matter_l_1',
            'select_matter_l_2',
            'answer_l_1',
            'answer_l_2',
            'answer_l_3',
            'best_result_system',
            'select_matter_system',
            'answer_system',
            'label_query',
            'label_answer',
            'label_fragments',
            'source_milana',
            'source_user_file',
            'source_web',
            'source_unknown',
            'web_search_available',
        )
        # Значения по умолчанию (английские)
        main.best_result_l_1 = 'Which of these fragments is most relevant to the query:"'
        main.best_result_l_2 = '"Return only the most suitable one: '
        main.request_l_1 = 'Query:'
        main.information_l_1 = 'Answer:'
        main.satisfies_l_1 = 'Fully satisfies?'
        main.select_matter_l_1 = 'Highlight the important. If the answer is already quite short, then just return the received answer unchanged. Query:'
        main.select_matter_l_2 = 'Answer:'
        main.answer_l_1 = 'Does the answer:'
        main.answer_l_2 = 'match the query'
        main.answer_l_3 = 'If the answer fully satisfies the query, output exactly "1". If the answer is incomplete or insufficient, output one or more reformulated queries (each on a new line) that would help find the missing information. Do not output any explanations or additional text.'

        main.best_result_system = main.best_result_l_1 + ' ' + main.best_result_l_2
        main.select_matter_system = main.select_matter_l_1
        main.answer_system = main.answer_l_1 + ' ' + main.answer_l_2 + '\n' + main.answer_l_3
        main.label_query = 'Query:'
        main.label_answer = 'Answer:'
        main.label_fragments = 'Fragments:'

        main.source_milana = 'System generated info'
        main.source_user_file = 'User files'
        main.source_web = 'Internet'
        main.source_unknown = 'Unknown source'
        main.web_search_available = ' (can also search on the internet)'
        return

    let_log('БИБЛИОТЕКАРЬ ВЫЗВАН')
    
    # Множество для отслеживания текстов запросов, по которым уже выполнялся веб-поиск
    web_search_done_texts = set()

    def find_engine(ask_text, attempts_left, donee='correct', search_target=1, full_output=True):
        nonlocal web_search_done_texts
        requests_to_libraries = [ask_text]
        embeddings = [get_embs(req) for req in requests_to_libraries]

        while attempts_left > 0:
            new_requests_to_libraries = []
            for i, emb in zip(requests_to_libraries, embeddings):
                let_log(i)
                # Список найденных элементов с источником
                items = []

                try:
                    if search_target == 1:  # milana_collection
                        base_filter = {}
                        if not global_state.gigo_web_search_allowed:
                            base_filter = {'source': {'$ne': 'web'}}

                        def query_collection(where_clause):
                            res_dict = coll_exec(
                                "query", "milana_collection",
                                query_embeddings=[emb],
                                filters=where_clause,
                                fetch=["documents", "metadatas"],
                                first=False
                            ) or {}
                            docs = res_dict.get('documents', []) or []
                            metas = res_dict.get('metadatas', []) or []
                            for meta, doc in zip(metas, docs):
                                if doc and doc.strip():
                                    items.append({
                                        'text': doc,
                                        'source': main.source_milana
                                    })

                        query_collection({"done": donee, "result": True})
                        query_collection({"done": donee})
                        if donee == 'correct':
                            query_collection({"done": 'incorrect'})

                    elif search_target == 2:  # user_collection
                        res_dict = coll_exec(
                            "query", "user_collection",
                            query_embeddings=[emb],
                            fetch=["documents", "metadatas"],
                            first=False
                        ) or {}
                        docs = res_dict.get('documents', []) or []
                        metas = res_dict.get('metadatas', []) or []
                        for meta, doc in zip(metas, docs):
                            if doc and doc.strip():
                                src = meta.get('source', '')
                                if src == 'web':
                                    source_str = main.source_web
                                elif src == 'file':
                                    fname = meta.get('name', 'unknown')
                                    source_str = f"{main.source_user_file} ({fname})"
                                else:
                                    source_str = main.source_unknown
                                items.append({'text': doc, 'source': source_str})

                except Exception as e:
                    let_log(f"[find_engine] Ошибка запроса: {e}")
                    items = []

                # Если ничего не найдено, пробуем веб-поиск
                if not items and global_state.gigo_web_search_allowed:
                    # Проверяем, не выполняли ли уже веб-поиск для этого текста запроса
                    if i not in web_search_done_texts:
                        try:
                            from cross_gpt import web_search, split_text_with_cutting, set_common_save_id, get_common_save_id
                            web_result = web_search(i)
                            if web_result and web_result != found_info_1:
                                # Сохраняем результат в user_collection
                                chunks = split_text_with_cutting(web_result)
                                if chunks:
                                    for t, chunk in enumerate(chunks):
                                        set_common_save_id()
                                        coll_exec(
                                            action="add",
                                            coll_name="user_collection",
                                            ids=[get_common_save_id()],
                                            embeddings=[get_embs(chunk)],
                                            metadatas=[{
                                                'name': i,
                                                'part': t + 1,
                                                'source': 'web'
                                            }],
                                            documents=[chunk]
                                        )
                                items.append({'text': web_result, 'source': main.source_web})
                                # Запоминаем, что для этого текста веб-поиск уже выполнен
                                web_search_done_texts.add(i)
                        except Exception as e:
                            let_log(f"Web search error: {e}")

                # Фильтруем пустые и "None"
                items = [it for it in items if it['text'].strip() and it['text'].strip().lower() != "none"]

                if items:
                    if full_output:
                        # Выбор лучшего результата
                        system_prompt = main.best_result_system
                        fragments = '\n'.join(it['text'] for it in items)
                        user_prompt = (main.label_query + ' ' + i + '\n' +
                                       main.label_fragments + '\n' + fragments)
                        try:
                            best_result = ask_model(user_prompt, system_prompt=system_prompt)
                        except:
                            best_result = ask_model(text_cutter(user_prompt), system_prompt=system_prompt)

                        if best_result and best_result.strip() and best_result != found_info_1:
                            # Ищем элемент с таким текстом
                            source_line = main.source_unknown
                            for it in items:
                                if it['text'] == best_result:
                                    source_line = it['source']
                                    break
                            if source_line == main.source_unknown and items:
                                source_line = items[0]['source']
                            return f"{source_line}\n{best_result}"
                        else:
                            return found_info_1

                    # Проверка на полное удовлетворение (берём первый элемент)
                    prompt_satisfies = (main.request_l_1 + '\n' + i + '\n' +
                                        main.information_l_1 + '\n' + items[0]['text'] + '\n' +
                                        main.satisfies_l_1)
                    answer_val = parse_prompt_response(prompt_satisfies, 1)
                    if answer_val == 1:
                        let_log('удовлетворяет')
                        return f"{items[0]['source']}\n{items[0]['text']}"

                    # Собираем полезные элементы
                    useful_items = []
                    for it in items:
                        prompt_answer = (main.answer_l_1 + '\n' + i + '\n' +
                                         main.answer_l_2 + '\n' + it['text'])
                        answer_val = parse_prompt_response(prompt_answer, 1)
                        if answer_val == 1:
                            useful_items.append(it)

                    # Выделение важного
                    if useful_items:
                        combined_text = '\n'.join(it['text'] for it in useful_items)
                        system_select = main.select_matter_system
                        user_select = (main.label_query + ' ' + i + '\n' +
                                       main.label_answer + '\n' + combined_text)
                        try:
                            selected_text = ask_model(user_select, system_prompt=system_select)
                        except:
                            selected_text = ask_model(text_cutter(user_select), system_prompt=system_select)
                        # Сохраняем источник от первого полезного элемента
                        if useful_items:
                            useful_items = [{'text': selected_text, 'source': useful_items[0]['source']}]

                    # Финальная оценка и генерация новых запросов
                    if useful_items:
                        combined_useful = '\n'.join(it['text'] for it in useful_items)
                        system_answer = main.answer_system
                        user_answer = (main.label_answer + ' ' + combined_useful + '\n' +
                                       main.label_query + ' ' + i)
                        try:
                            answer = ask_model(user_answer, system_prompt=system_answer)
                        except:
                            answer = ask_model(text_cutter(user_answer), system_prompt=system_answer)

                        if _extract_first_digit(answer, 0) != 0:
                            # Возвращаем все полезные элементы с источниками
                            return '\n\n'.join(f"{it['source']}\n{it['text']}" for it in useful_items)
                        else:
                            new_requests_to_libraries.extend(answer.splitlines())
                    # Если полезных нет, но были элементы – ничего не добавляем (цикл продолжится)
                # else: items пуст – ничего не добавляем

            if new_requests_to_libraries:
                requests_to_libraries = new_requests_to_libraries
                embeddings = [get_embs(req) for req in requests_to_libraries]
                attempts_left -= 1
            else:
                break

        return found_info_1

    def process_single_question(quest):
        # Обработка входящего запроса для одиночного вопроса
        first_request_target = 1
        full_output = False
        if quest and quest[0] in ('1', '2', '3'):
            first_request_target = int(quest[0])
            if len(quest) > 1 and quest[1] == '1':
                full_output = True
                quest = quest[2:]
            elif len(quest) > 1 and quest[1] == '2':
                quest = quest[2:]
            else:
                quest = quest[1:]

        result_primary = find_engine(quest, global_state.librarian_max_attempts,
                                     search_target=1, full_output=full_output) or ""
        result_user = find_engine(quest, global_state.librarian_max_attempts,
                                  search_target=2, full_output=full_output) or ""

        parts = [result_primary, result_user]
        final_res = ''
        for p in parts:
            if p and p != found_info_1 and p.lower() != "none":
                final_res += p + '\n\n---\n\n'
        if not final_res:
            final_res = found_info_1
        return final_res.strip()

    if isinstance(quest, str) and '\n' in quest:
        lines = quest.splitlines()
        question_lines = [line for line in lines if '?' in line]
        if not question_lines:
            question_lines = lines
        cleaned_questions = []
        for question in question_lines:
            cleaned = re.sub(r'^\s*(?:\d+[\.\)]\s*|[-*•]\s*)*', '', question.strip())
            if cleaned:
                cleaned_questions.append(cleaned)
        additional_info = ''
        for question in cleaned_questions:
            answer = process_single_question(question)
            if answer != found_info_1:
                additional_info += '\n' + answer
        return additional_info.strip() if additional_info else found_info_1
    else:
        return process_single_question(quest)