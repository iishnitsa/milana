'''
new_dialog
passes the task for decomposition into subtasks and execution
'''

# Токены
from cross_gpt import (
    slash_token,
    slash_n,
    zpt_token,
    prompt_n,
    zpt_space,
    space_skb,
    closing_skb,
    milana_template,
    system_role_text,
    operator_role_text,
    func_role_text,
    wrong_command,
    start_dialog_history,
    let_log
)

def find_tuple_by_first_list(data, target_list):
    for list1, list2, obj in data:
        if list1 == target_list:
            return (list1, list2), obj
    return None  # Если совпадение не найдено

def main(client_task):
    # Инициализация атрибутов модуля при первом вызове
    if not hasattr(main, 'attr_names'):
        let_log('dialog initialization start')
        main.attr_names = (
            'start_dialog_tool_text_1',
            'start_dialog_tool_text_2',
            'operator_tool_prompt'
        ) # TODO: нужно не грузить промпт для оператора а только задачу
        main.start_dialog_tool_text_1 = '''You're reading a task description for an operator who will supervise a specialist's work. Based on the task description, select tools from the list of allowed tools.

Output requirements:

Only tool names separated by comma and space in a single line.
No quotes, no trailing dots, no explanations, and no additional characters.
If no tools are suitable or if any incorrect or extra names are found, output exactly: None

List of allowed tools:

'''
        main.start_dialog_tool_text_2 = '''

RESTRICTIONS:

Prohibited to output any tool names other than those from the list.
Creating new commands is forbidden.
Any typo or incorrect entry in the list is sufficient reason to output None.

Task description:

'''
        main.operator_tool_prompt = '\nTo call a tool (only one call per response), you need to start your response with three exclamation marks, then the tool name, then three more exclamation marks, followed by the information for the tool. The command must be sent as-is, without interpretation, for example: "!!!end_dialog!!!Task answer: 4".\nYou only have access to the following tools: '
        return
    let_log('начинается диалог')
    #let_log(start_dialog_history)
    from cross_gpt import (
        global_state,
        ask_model,
        text_cutter,
        sql_exec,
        gigo,
        tools_selector,
        create_new_specialist,
        down_now_try,
        traceprint,
    )
    import difflib

    global_state.stop_agent = True
    global_state.need_start_new_dialog = True
    if client_task == '':
        prompt = gigo(global_state.main_now_task)
        global_state.retries = False
    elif global_state.task_retry:
        global_state.main_now_task = client_task
        global_state.retries.append(0)
        prompt = gigo(client_task)
    else:
        prompt = gigo(client_task)

    # Выбор инструментов для Миланы
    # Выбор инструментов для Миланы
    milana_tools = global_state.milana_module_tools
    if global_state.module_tools_keys: # TODO: разберись с этим промптом
        need_tools_raw = ask_model(
            main.start_dialog_tool_text_1 +
            global_state.tools_str +
            main.start_dialog_tool_text_2 +
            prompt
        )
        let_log(need_tools_raw)
        split_tools = [t.strip().replace('\\_','_') for t in need_tools_raw.split(',') if t.strip()]
        unique_tools = list(set(split_tools))
        for nt in unique_tools:
            for tool_text, (tool_tokens) in global_state.an_t_str.items():
                if nt in tool_text or difflib.get_close_matches(nt, [tool_text], n=1, cutoff=0.85):
                    #milana_tools[tool_tokens] = find_tuple_by_first_list(global_state.another_tools, tool_tokens)#[1:]
                    res = find_tuple_by_first_list(global_state.another_tools, tool_tokens)
                    if res is not None:
                        (list1, list2), func = res
                        milana_tools[tool_tokens] = (list2, func)
                        break
        let_log('ошибки нет')
    # Формирование финального промпта
    full_prompt = milana_template + prompt + main.operator_tool_prompt
    let_log(milana_tools)
    for tool in milana_tools:
        full_prompt += tool + space_skb + milana_tools[tool][0] + closing_skb + zpt_token
    let_log(full_prompt)
    let_log(global_state.another_tools)
    if full_prompt and full_prompt[-1] == zpt_token:
        full_prompt = full_prompt[:-1]
    
    global_state.conversations += 1
    # TODO: ЗАПИСЫВАЕТСЯ КАКАЯ-ТО ФИГНЯ МИЛАНЕ
    # СИСТЕМ ТЭГ ПРОВАЙДЕР СДЕЛАЙ ЧТОБЫ ГЛОБАЛЬНО ГРУЗИЛ И СЮДА И ВЕЗДЕ
    # ПОПРОСИ ПИСАТЬ КРАТКО (ВООБЩЕ СОКРАЩЕНИЕ ЭТО СЛОЖНАЯ ТЕМА)
    # МОЖЕТ РАЗБИТИЕ НА ОЧЕНЬ МАЛЕНЬКИЕ ТЕЗИСЫ
    # ИЗБАВЬСЯ ОТ KPI
    sql_exec(
        'INSERT INTO chats (chat_id, prompt, history) VALUES (?, ?, ?)',
        (global_state.conversations, full_prompt, '')
    )
    global_state.tools_commands_dict[global_state.conversations] = milana_tools

    # Генерация начального ответа
    try:
        talk_prompt = ask_model(
            system_role_text +
            full_prompt +
            operator_role_text
        )
    except:
        try:
            talk_prompt = ask_model(
                system_role_text +
                text_cutter(full_prompt) +
                operator_role_text
            )
        except:
            raise
    history = operator_role_text + talk_prompt
    sql_exec(
        "UPDATE chats SET history = ? WHERE chat_id = ?",
        (history, global_state.conversations)
    )
    
    let_log("НАЧАЛЬНЫЙ ОТВЕТ МИЛАНЫ:")
    let_log(talk_prompt)
    answer = tools_selector(talk_prompt, global_state.conversations) # принт что-ли какой-то
    if answer and answer != wrong_command: # неправильная команда на 7b q3
        talk_prompt = answer
    else:
        let_log('СОЗДАНИЕ НОВОГО СПЕЦИАЛИСТА...')
        talk_prompt = create_new_specialist(talk_prompt)

    let_log("ОТВЕТ ПОСЛЕ ОБРАБОТКИ:")
    let_log(talk_prompt)
    # ЧТО ЭТО
    last_talk_prompt = talk_prompt
    try:
        talk_prompt = ask_model(
            system_role_text +
            full_prompt +
            history +
            func_role_text +
            talk_prompt +
            operator_role_text
        )
    except:
        history = start_dialog_history + text_cutter(history)
        try:
            talk_prompt = ask_model(
                system_role_text +
                full_prompt +
                history +
                func_role_text +
                talk_prompt +
                operator_role_text
            )
        except Exception as e:
            try:
                talk_prompt = ask_model(
                    system_role_text +
                    full_prompt +
                    history +
                    func_role_text +
                    text_cutter(talk_prompt) +
                    operator_role_text
                )
            except: raise
    history += func_role_text + last_talk_prompt + operator_role_text + talk_prompt
    sql_exec(
        "UPDATE chats SET history = ? WHERE chat_id = ?",
        (history, global_state.conversations - 1)
    )

    down_now_try()
    return talk_prompt
