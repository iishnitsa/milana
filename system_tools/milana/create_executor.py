'''
create_executor
receives a task, creates an executor and, possibly, provides them with tools suitable for solving this task
'''

from cross_gpt import (
    found_info_1,
    find_all_commands,
    only_one_func_text,
    what_is_func_text,
    native_func_call,
    let_log,
    global_state,
    ask_model,
    text_cutter,
    sql_exec,
    save_emb_dialog,
    librarian,
    next_executor,
    chunk_size,
    parse_prompt_response,
    system_role_text,
    create_chat,
    delete_chat,
    get_chat_context,
    gigo_questions,
    get_level,
    prompt_evaluation_2,
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
            'worker_base',
            'worker_delegation_part',
            'command_example',
            'avaiable_tools_text',
            'create_executor_return_text_1',
            'create_executor_return_text_2',
            'hierarchy_limit_info',
            'delegate_unavailable_for_executor',
            'need_info_example',
        )

        main.create_executor_param_1 = 'Are the tasks the same? Task 1:\n'
        main.create_executor_param_2 = '\nTask 2:\n'
        main.create_executor_questions = 'Write questions, separating them with ; to search for additional information for this task:\n'
        main.additional_info_text = 'Additional information:\n'
        main.create_executor_write_prompt_1 = '''
Write instructions and hints for the AI executor based on this task. Describe what he should and can do, having only the following tools.
IMPORTANT: The generated instructions will be ADDED to the executor's system prompt. Therefore, do NOT include any role declarations ("You are an AI executor..."), greetings, or generic phrases. Focus solely on the specific task, steps, and guidelines.
'''
        main.create_executor_write_prompt_2 = '''
Write instructions and hints for the AI executor based on this task. Describe what he should and can do.
IMPORTANT: The generated instructions will be ADDED to the executor's system prompt. Therefore, do NOT include any role declarations ("You are an AI executor..."), greetings, or generic phrases. Focus solely on the specific task, steps, and guidelines.
'''
        main.create_executor_select_tools_1 = '''You will receive a task description from the user. Based on the task description, select tools from the list of allowed tools.
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
'''
        main.worker_base = '''
You are an AI executor "Ivan".
You work with your curator - AI operator "Milana". Perform the task assigned by "Milana". She monitors the execution.
Perform the task, discussing each step with "Milana", listening to her comments.
'''
        main.worker_delegation_part = '''
Only if the task is too complex and extensive, or only if "Milana" is dissatisfied with the result after several attempts and hard work, call the task delegation command and describe the task (in detail, with unambiguous interpretation, with explanation of abbreviations if any), problems, and the reason for dissatisfaction if any.
This will create a similar dialog of "Milana" and "Ivan" lower in the hierarchy, create a plan with subtasks based on this task, and pass this plan to them.
In response to the delegation command, you will receive only the result or a failure message.
'''
        main.command_example = '''
Example of a command call - "!!!delegate_task!!! Implement a warehouse accounting system for an online store"
'''
        main.need_info_example = '''
Example of a command call - "!!!need_information!!! Documentation for React hooks"
'''
        main.avaiable_tools_text = 'Available tools:'
        main.create_executor_return_text_1 = "Executor has been created, welcome him (write a welcome message, don't write a command)"
        main.create_executor_return_text_2 = "Executor has been recreated, welcome the new executor(write a welcome message, don't write a command)"
        main.hierarchy_limit_info = 'Hierarchy levels are limited. Current level'
        main.delegate_unavailable_for_executor = 'The task delegation function down the hierarchy is not available.'
        return
    # Определяем, пересоздание это или первое создание
    if global_state.conversations % 2 == 0: # ПЕРЕСОЗДАНИЕ исполнителя
        return_text = main.create_executor_return_text_2
        let_log('ПЕРЕСОЗДАНИЕ ИСПОЛНИТЕЛЯ')
        lt = global_state.last_task_for_executor.get(global_state.conversations, '')
        if text != '' and text is not None: param = parse_prompt_response(main.create_executor_param_1 + text + main.create_executor_param_2 + lt, 0)
        else: param = 0
        if param == 1: tag = 'correct'
        else: tag = 'incorrect'
        let_log(f"Пересоздание исполнителя, тег='{tag}' (param={param}, задача {'та же' if param == 0 else 'разная'})")
        save_emb_dialog(tag, 'executor')
        let_log(f"Сохранили старого исполнителя с тегом '{tag}'")
        delete_chat(global_state.conversations)
        let_log("Удалили старый чат исполнителя")
    else: # ПЕРВОЕ СОЗДАНИЕ исполнителя на этом уровне
        return_text = main.create_executor_return_text_1
        global_state.conversations += 1
        let_log('создание нового специалиста')
    global_state.last_task_for_executor[global_state.conversations] = text
    next_executor()
    # Получаем дополнительную информацию через библиотекаря
    questions_raw = ask_model(text, system_prompt=gigo_questions)
    additional_info = librarian(questions_raw)
    if additional_info != found_info_1:
        additional_info = main.additional_info_text + additional_info
    else:
        additional_info = ''
        let_log("Библиотекарь не нашел дополнительной информации")
    ivan_tools = global_state.ivan_module_tools.copy()
    
    # Определяем, может ли этот исполнитель делегировать
    current_level = get_level()
    if global_state.hierarchy_limit == 0:
        delegation_allowed = True
    else:
        delegation_allowed = current_level < global_state.hierarchy_limit
    # Удаляем команду делегирования, если недоступна
    if global_state.hierarchy_limit == 1 and global_state.start_dialog_command_name in ivan_tools:
        del ivan_tools[global_state.start_dialog_command_name]
        let_log("Удалена команда делегирования из инструментов исполнителя")
    
    # Выбор инструментов для исполнителя
    if global_state.module_tools_keys:
        let_log('есть модуль тулз киз')
        need_tools_raw = ask_model(
            text,
            system_prompt=main.create_executor_select_tools_1 +
                          global_state.tools_str +
                          main.create_executor_select_tools_2
        )
        let_log('Результат выбора инструментов:')
        let_log(need_tools_raw)
        tools_names = find_all_commands(need_tools_raw, global_state.module_tools_keys)
        let_log(f"Найдены инструменты: {tools_names}")
        for name in tools_names:
            for tool_tokens, tool_desc, tool_func in global_state.another_tools:
                if name == tool_tokens:
                    ivan_tools[tool_tokens] = (tool_desc, tool_func)
                    let_log(f"Добавлен инструмент: {tool_tokens}")
                    break
    # Формируем строку с описанием инструментов для промпта
    selected_ivan_tools = ''
    for tool in ivan_tools: selected_ivan_tools += tool + ' (' + ivan_tools[tool][0] + ')\n'
    # Определяем, какой промпт использовать (с инструментами или без)
    if selected_ivan_tools:
        system_prompt_for_instructions = main.create_executor_write_prompt_1
        user_content = text + additional_info + f"\n\n{main.avaiable_tools_text}\n" + selected_ivan_tools
    else:
        system_prompt_for_instructions = main.create_executor_write_prompt_2
        user_content = text + additional_info
    # Генерируем инструкции для исполнителя на основе задачи
    let_log("Генерация инструкций для исполнителя...")
    instructions = ask_model(
        user_content,
        system_prompt=system_prompt_for_instructions
    )
    # Формируем финальный промпт для исполнителя
    prompt = main.worker_base + prompt_evaluation_2 + ' ' + text
    if global_state.hierarchy_limit != 1:
        prompt += main.worker_delegation_part

    # Добавляем информацию об иерархии, если лимит больше 1
    hierarchy_note = ""
    if global_state.hierarchy_limit > 1:
        limit = global_state.hierarchy_limit
        hierarchy_note = f"\n{main.hierarchy_limit_info} {current_level}/{limit}.\n"
        if not delegation_allowed:
            hierarchy_note += f"\n{main.delegate_unavailable_for_executor}\n"
    prompt += hierarchy_note

    prompt += instructions + only_one_func_text
    if ivan_tools:
        prompt += selected_ivan_tools

    # Добавляем пример вызова команды для запроса информации (всегда, если не нативный вызов)
    if not native_func_call:
        prompt += what_is_func_text + main.need_info_example

    # Сохраняем инструменты для этого чата
    global_state.tools_commands_dict[global_state.conversations] = ivan_tools
    let_log('ДОСТУПНЫЕ ИНСТРУМЕНТЫ ДЛЯ ИСПОЛНИТЕЛЯ:')
    for tool, (desc, _) in ivan_tools.items(): let_log(f"  {tool}: {desc}")
    # Создаем чат для исполнителя через менеджер
    system_prompt = system_role_text + prompt
    create_chat(global_state.conversations, system_prompt)
    return return_text