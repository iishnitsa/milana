'''
create_executor
receives a task, creates an executor and, possibly, provides them with tools suitable for solving this task
'''

# Получаем текстовые константы из контейнера
from cross_gpt import (
    slash_token,
    dot_zpt_token,
    slash_n,
    one_token,
    two_token,
    three_token,
    zpt_token,
    space_skb,
    closing_skb,
    prompt_n,
    found_info_1,
    zpt_space,
    find_all_commands,
    what_is_func_text,
    native_func_call,
    let_log
)

def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'create_executor_param_1',
            'create_executor_param_2',
            'create_executor_questions',
            'additional_info_text',
            'create_executor_write_prompt_1',
            'create_executor_write_prompt_2',
            'create_executor_select_tools_1',
            'create_executor_select_tools_2',
            'worker_tool_prompt_1',
            'worker_tool_prompt_2',
            'create_executor_return_text_1',
            'create_executor_return_text_2',
        )

        main.create_executor_param_1 = 'Are the tasks the same? If they are different, send "1", if they are the same - "0". Task 1:\n'
        main.create_executor_param_2 = '\nTask 2:\n'
        main.create_executor_questions = 'Write questions, separating them with ; to search for additional information for this task:\n'
        main.additional_info_text = 'Additional information:\n'
        main.create_executor_write_prompt_1 = '\nWrite instructions and prompts for the AI executor based on this task. Describe what it should and can do, having only the following tools:\n'
        main.create_executor_write_prompt_2 = '\nWrite instructions and prompts for the AI executor based on this task. Describe what it should and can do:\n'
        main.create_executor_select_tools_1 = '''You read the description of a task for an executor. Based on the task description, select tools from the list of allowed tools.

Output data:

Only tool names separated by comma and space in a single line.
No quotes, no periods at the end, no explanations, and no additional characters.
If no tool is suitable or if at least one incorrect or extra name is found, output exactly: None

List of allowed tools:

'''

        main.create_executor_select_tools_2 = '''

LIMITATIONS:

It is forbidden to output any other tool names except those from the list.
You cannot invent new commands.
Any typo or incorrect entry in the list is sufficient reason to output None.

Task description:

'''
        main.worker_tool_prompt_1 = '''
You are the AI executor "Ivan".
You work with your curator - the AI operator "Milana". Perform the task assigned by "Milana". She controls the execution.
Perform the task, discussing each step with "Milana", listening to her comments.
Only if the task is too complex and extensive, or only if "Milana" is dissatisfied with the result after several attempts and hard work, call the task delegation command and after the command, describe the task, problems, and the reason for dissatisfaction, if any.
This will create a similar dialogue between "Milana" and "Ivan" further down the hierarchy, create a plan with subtasks based on this task, and transfer this plan to them.
In response to the delegation command, you will only receive the result or a failure message.

'''
        main.worker_tool_prompt_2 = '''

To call a command, write three exclamation marks at the beginning of the message, then the command name, then three more exclamation marks, and then the information for the command.
Example of a command call - "!!!delegate_task!!! Implement a warehouse accounting system for an online store"
You can only call the commands available below. Their description of work is in parentheses:

'''
        main.create_executor_return_text_1 = "Executor has been created, welcome him (write a welcome message, don't write a command)"
        main.create_executor_return_text_2 = "Executor has been recreated, welcome the new executor(write a welcome message, don't write a command)"
        return

    # НОВЫЕ ИМПОРТЫ ИЗ CHAT_MANAGER
    from chat_manager import create_chat, delete_chat, get_chat_context
    from cross_gpt import (
        global_state,
        ask_model,
        text_cutter,
        get_embs,
        milana_collection,
        sql_exec,
        save_emb_dialog,
        librarian,
        set_now_try,
        get_now_try,
        chunk_size,
        parse_prompt_response,
        system_role_text,
    )
    import difflib
    set_now_try()

    if global_state.conversations % 2 == 0: # ОН ДОЛЖЕН УДАЛИТЬ ЧАТ С ПРОШЛЫМ СПЕЦИАЛИСТОМ
        return_text = main.create_executor_return_text_1
        let_log('ПЕРЕСОЗДАНИЕ специалиста')
        lt = global_state.last_task_for_executor[global_state.conversations]
        if text != '' and text is not None:
            param = parse_prompt_response(main.create_executor_param_1 + 
                text + 
                main.create_executor_param_2 +
                lt, 0)
        else: param = 0
        parts = get_now_try().split(slash_token) # может тут из-за токена ошибка была?
        parents_id = slash_token.join(parts[:-1]) if len(parts) > 1 else slash_token

        sql_exec(
            'INSERT INTO tries (try_id, task, parents_id) VALUES (?, ?, ?)',
            (parts[-1], global_state.main_now_task, parents_id)
        )

        # ИЗМЕНЕНИЕ: Получаем историю и удаляем чат через менеджер
        _, history = get_chat_context(global_state.conversations)
        if history:
            save_emb_dialog(history, 'incorrect' if param == 0 else 'correct') # TODO:
        
        delete_chat(global_state.conversations)

    else:
        return_text = main.create_executor_return_text_1
        global_state.conversations += 1
        print('создание нового специалиста')
        global_state.last_task_for_executor[global_state.conversations] = text

    questions_raw = ask_model(main.create_executor_questions + text, all_user=True)
    additional_info = librarian(questions_raw)
    if additional_info != found_info_1:
        #while len(additional_info) > chunk_size:
        #    text_cutter(additional_info)
        additional_info = (
            main.additional_info_text +
            additional_info
        )
    else: additional_info = ''
    
    ivan_tools = global_state.ivan_module_tools
    # добавь условие когда нет тулз
    # и если их нет тогда что?
    # должны же быть базовые
    # но в теории если совсем нет то и в промпте не должно быть
    # указано использование команд
    # а зачем тогда иф если должны быть базовые
    # типа дополнительные, а где базовые тогда
    if global_state.module_tools_keys:
        let_log('есть модуль тулз киз')
        prompt_tools = (
            main.create_executor_select_tools_1 +
            global_state.tools_str +
            main.create_executor_select_tools_2 +
            text
        )
        need_tools_raw = ask_model(prompt_tools, all_user=True)
        all_tool_names = [tool[0] for tool in global_state.another_tools]
        # Новый участок — замена старой логики
        let_log('ВОТ')
        let_log(need_tools_raw)
        tools_names = find_all_commands(need_tools_raw, all_tool_names)
        
        for name in tools_names:
            # Ищем инструмент по точному имени в исходном списке
            for tool_tokens, tool_desc, tool_func in global_state.another_tools:
                if name == tool_tokens:
                    ivan_tools[tool_tokens] = (tool_desc, tool_func)
                    break
    selected_ivan_tools = ''
    for tool in ivan_tools: selected_ivan_tools += tool + space_skb + ivan_tools[tool][0] + closing_skb + zpt_token + ' '
    if selected_ivan_tools and selected_ivan_tools[-1] == zpt_token: selected_ivan_tools = selected_ivan_tools[:-1]
    if selected_ivan_tools: what_write_prompt = main.create_executor_write_prompt_1
    else: what_write_prompt = main.create_executor_write_prompt_2
    
    # В файле cross_gpt.py в `worker_tool_prompt_1` я заметил переменную `generated_executor_instructions`.
    # Поскольку её определения нет в этом файле, я закомментирую её, чтобы избежать ошибки.
    # Если она должна быть здесь, раскомментируйте и убедитесь, что она определена.
    # prompt = main.worker_tool_prompt_1 + generated_executor_instructions + ask_model(
    prompt = main.worker_tool_prompt_1 + ask_model(
        text +
        additional_info +
        what_write_prompt +
        selected_ivan_tools, all_user=True
    )
    if ivan_tools: prompt += main.worker_tool_prompt_2 + selected_ivan_tools
    if not native_func_call: prompt += what_is_func_text
    
    global_state.tools_commands_dict[global_state.conversations] = ivan_tools

    let_log('ДОСТУПНЫЕ ИНСТРУМЕНТЫ:')
    let_log(global_state.tools_commands_dict)
    
    # ИЗМЕНЕНИЕ: Создаем чат через менеджер
    create_chat(global_state.conversations, system_role_text + prompt)

    #global_state.stop_agent = True # TODO: что это???
    return return_text