'''
need_information
call to get missing information
'''

from cross_gpt import (
    global_state,
    ask_model,
    text_cutter,
    get_embs,
    coll_exec,
    let_log,
    found_info_1,
)
import re

def main(quest): # TODO: нужно удалять дубликаты информации
    # Инициализация атрибутов модуля при первом вызове
    if not hasattr(main, 'attr_names'):
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
            'answer_l_4'
        )
        main.best_result_l_1 = 'Which of these fragments is most relevant to the query:"'
        main.best_result_l_2 = '"Return only the most suitable one: '
        main.request_l_1 = 'Query:'
        main.information_l_1 = 'Answer:'
        main.satisfies_l_1 = 'Fully satisfies? "1" - yes, no - "0"'
        main.select_matter_l_1 = 'Highlight the important. If the answer is already quite short, then just return the received answer unchanged. Query:'
        main.select_matter_l_2 = 'Answer:'
        main.answer_l_1 = 'Does the answer:'
        main.answer_l_2 = 'match the query'
        main.answer_l_3 = 'if yes, send "1", otherwise we received one or more additional reformulated requests for additional information'
        main.answer_l_4 = '"1" - yes, "0" - no'
        return
    let_log('БИБЛИОТЕКАРЬ ВЫЗВАН')
    def find_engine(ask_text, attempts_left, donee='correct', search_target=1, full_output=True):
        requests_to_libraries = [ask_text]
        complex_result = ''
        embeddings = [get_embs(req) for req in requests_to_libraries]
        while attempts_left > 0:
            new_requests_to_libraries = []
            maybe_new_requests_to_libraries = []
            for i, emb in zip(requests_to_libraries, embeddings):
                let_log(i)
                results = []
                try:
                    if search_target == 1:
                        base_filter = {}
                        if not global_state.gigo_web_search_allowed:
                            base_filter = {'source': {'$ne': 'web'}}
                        # Копируем базовый фильтр и добавляем специфичные условия
                        where1 = base_filter.copy()
                        where1.update({"done": donee, "result": True})
                        where2 = base_filter.copy()
                        where2.update({"done": donee})
                        where3 = base_filter.copy()
                        if donee == 'correct': where3 = {"done": 'incorrect'}
                        results = coll_exec(
                            "query",
                            "milana_collection",
                            query_embeddings=[emb],
                            filters=where1,
                            fetch="documents",
                            first=False
                        ) or []
                        results += coll_exec(
                            "query",
                            "milana_collection",
                            query_embeddings=[emb],
                            filters=where2,
                            fetch="documents",
                            first=False
                        ) or []
                        if where3:
                            results += coll_exec(
                                "query",
                                "milana_collection",
                                query_embeddings=[emb],
                                filters=where3,
                                fetch="documents",
                                first=False
                            ) or []
                    elif search_target == 2:
                        results = coll_exec(
                            "query",
                            "user_collection",
                            query_embeddings=[emb],
                            fetch="documents",
                            first=False
                        ) or []
                    elif search_target == 3: pass
                except Exception as e: let_log(f"[find_engine] Ошибка запроса: {e}")
                # Если ничего не найдено, пробуем поиск в интернете
                if (not results or results == [None]) and global_state.gigo_web_search_allowed:
                    try:
                        from cross_gpt import web_search, split_text_with_cutting
                        web_result = web_search(i)
                        if web_result and web_result != found_info_1:
                            # Сохраняем результат в user_collection
                            chunks = split_text_with_cutting(web_result)
                            if chunks:
                                for t, chunk in enumerate(chunks):
                                    # Генерируем уникальный ID
                                    import hashlib # TODO: избавься от этого
                                    chunk_id = hashlib.md5(f"{i}_{t}".encode()).hexdigest()
                                    coll_exec(
                                        action="add",
                                        coll_name="user_collection",
                                        ids=[chunk_id],
                                        embeddings=[get_embs(chunk)],
                                        metadatas=[{
                                            'name': i,  # текст запроса
                                            'part': t + 1,
                                            'source': 'web'
                                        }],
                                        documents=[chunk]
                                    )
                            results = [web_result]
                    except Exception as e: let_log(f"Web search error: {e}")
                let_log(results)
                if results and results != [None]:
                    if full_output:
                        try: 
                            best_result = ask_model(
                                main.best_result_l_1 + '\n' + i + '\n' +
                                main.best_result_l_2 + '\n' + '\n'.join(results), all_user=True
                            )
                        except: 
                            best_result = ask_model(
                                main.best_result_l_1 + '\n' + i + '\n' +
                                main.best_result_l_2 + '\n' + text_cutter('\n'.join(results)), all_user=True
                            )
                        return best_result # TODO: нужно условие что ничего не найдено
                    answer = ask_model(
                        main.request_l_1 + '\n' + i + '\n' +
                        main.information_l_1 + '\n' + results[0] + '\n' +
                        main.satisfies_l_1, all_user=True
                    )
                    try:
                        if '1' in answer[0:3]:
                            let_log('удовлетворяет')
                            return results[0]
                    except: pass
                    for r in results:
                        let_log('убрал сокращение каждого результата перед проверкой на полезность')
                        try: 
                            answer = ask_model(
                                main.answer_l_1 + '\n' + i + '\n' +
                                main.answer_l_2 + '\n' + r + '\n' +
                                main.answer_l_4, all_user=True
                            )
                        except:
                            answer = ask_model(
                                main.answer_l_1 + '\n' + i + '\n' +
                                main.answer_l_2 + '\n' + text_cutter(r) + '\n' +
                                main.answer_l_4, all_user=True
                            )
                        try:
                            if '1' in answer[0:3]: complex_result += r + '\n'
                        except: pass
                    try:
                        complex_result = ask_model(
                            main.select_matter_l_1 + '\n' + i + '\n' +
                            main.select_matter_l_2 + '\n' + complex_result, all_user=True
                        )
                    except:
                        complex_result = ask_model(
                            main.select_matter_l_1 + '\n' + i + '\n' +
                            main.select_matter_l_2 + '\n' + text_cutter(complex_result), all_user=True
                        )
                    answer = ask_model(
                        main.answer_l_1 + '\n' + complex_result + '\n' +
                        main.answer_l_2 + '\n' + i + '\n' +
                        main.answer_l_3, all_user=True
                    )
                    try:
                        if '1' in answer[0:12]: return complex_result
                        else: new_requests_to_libraries.extend(answer.splitlines())
                    except: pass
            if new_requests_to_libraries:
                requests_to_libraries = new_requests_to_libraries
                embeddings = [get_embs(req) for req in requests_to_libraries]
                attempts_left -= 1
            else: break
        if not complex_result: return found_info_1
        return complex_result

    def process_single_question(quest):
        # Обработка входящего запроса для одиночного вопроса
        first_request_target = 1
        full_output = False
        if quest and quest[0] in ('1', '2', '3'):
            first_request_target = quest[0]
            if len(quest) > 1 and quest[1] == '1':
                full_output = True
                quest = quest[2:]
            elif len(quest) > 1 and quest[1] == '2': quest = quest[2:]
            else: quest = quest[1:]
        # Первый проход — milana_collection
        result_primary = find_engine(
            quest,
            global_state.librarian_max_attempts,
            search_target=1,
            full_output=full_output
        ) or ""
        # Второй проход — user_collection
        result_user = find_engine(
            quest,
            global_state.librarian_max_attempts,
            search_target=2,
            full_output=full_output
        ) or ""
        parts = [result_primary, result_user]
        def is_valid(x):
            if not x: return False
            if x.lower() == "none": return False
            if x == found_info_1: return False
            return True
        final_res = ''
        for p in parts:
            if is_valid(p): final_res += p + '\n'
        if final_res == '': final_res = found_info_1
        return final_res
    if isinstance(quest, str) and '\n' in quest:
        lines = quest.splitlines()
        question_lines = [line for line in lines if '?' in line]
        if not question_lines: question_lines = lines
        cleaned_questions = []
        for question in question_lines:
            # Убираем нумерацию вида "1. ", "2) " и т.п. в начале строки
            cleaned = re.sub(r'^\s*\d+[\.\)]\s*', '', question.strip())
            if cleaned: cleaned_questions.append(cleaned)
        # Обрабатываем каждый вопрос отдельно
        additional_info = ''
        for question in cleaned_questions:
            answer = process_single_question(question)
            let_log(found_info_1)
            if answer != found_info_1: additional_info += '\n' + answer
        return additional_info.strip() if additional_info else found_info_1
    else: return process_single_question(quest)