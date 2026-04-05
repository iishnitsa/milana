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
    no_markdown_instruction,
    write_shortly_prompt,
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
            'avaiable_tools_text',
            'create_executor_return_text_1',
            'create_executor_return_text_2',
            'hierarchy_limit_info',
            'delegate_unavailable_for_executor',
            'need_info_example',
            'tasks_identical_text',
            'exec_anti_loop_text',
        )
        main.create_executor_param_1 = 'Are the tasks the same?'
        main.create_executor_param_2 = 'Task'
        main.create_executor_questions = 'Write questions, separating them with ; to search for additional information for this task:\n'
        main.additional_info_text = 'Additional information:\n'
        main.create_executor_write_prompt_1 = '''
Write a concise instruction (one short paragraph) for the AI executor (Ivan) based on the task. This instruction will be added to his system prompt.

WHAT SHOULD BE IN THE INSTRUCTION:
- Explain the overall goal — what exactly needs to be done.
- Describe how he can use the available tools (list below) to accomplish subtasks. In what situations might each tool be useful?
- Remind him that he can ONLY use these tools — no other commands are allowed.

WHAT NOT TO INCLUDE:
- DO NOT add role declarations ("You are an AI executor Ivan..."), greetings, or generic phrases — they are already in the base prompt.
- DO NOT give a detailed step-by-step plan — the operator (Milana) will guide the process in the dialogue.
- DO NOT write examples of command syntax ("!!!command!!! ...") — the correct format is already known from the product.

Focus on WHAT to do and HOW the tools help, not on command syntax.
'''
        main.create_executor_write_prompt_2 = '''
Write a concise instruction (one short paragraph) for the AI executor (Ivan) based on the task. This instruction will be added to his system prompt.

WHAT SHOULD BE IN THE INSTRUCTION:
- Explain the overall goal — what exactly needs to be done.
- If any tools might be useful, mention them conceptually (even if a specific list is not provided).
- Remind him that he is limited to the commands available in the system (cannot invent his own).

WHAT NOT TO INCLUDE:
- DO NOT add role declarations ("You are an AI executor Ivan..."), greetings, or generic phrases.
- DO NOT give a detailed step-by-step plan — the operator (Milana) will guide the process.
- DO NOT write examples of command syntax.

Focus on WHAT to do and how to move towards the goal.
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
You are an AI executor "Ivan". Perform tasks assigned by the curator — AI operator "Milana". Discuss each step, listen to comments. She monitors the execution.
'''
        main.worker_delegation_part = '''
Only if the task is too complex and extensive, or if "Milana" is dissatisfied with the result after several attempts and hard work, call the task delegation command and describe the task in detail (with unambiguous interpretation, explanation of abbreviations if any), the problems, and the reason for dissatisfaction if any.
This will create a similar "Milana" and "Ivan" dialog lower in the hierarchy, create a plan with subtasks based on this task, and pass it to them.
In response to the delegation command, you will receive only the result or a failure message.
'''
        main.need_info_example = '''
Example command call - "!!!need_info!!! React hooks documentation"
'''
        main.avaiable_tools_text = 'Available tools:'
        main.create_executor_return_text_1 = 'Executor has been created.'
        main.create_executor_return_text_2 = 'Executor has been recreated.'
        main.hierarchy_limit_info = 'Hierarchy levels are limited. Current level'
        main.delegate_unavailable_for_executor = 'Task delegation down the hierarchy is not available.'
        main.tasks_identical_text = 'Tasks are absolutely identical. If you are consciously recreating the executor with the same task, and if it makes sense, the task must differ by at least 1 character.'
        main.exec_anti_loop_text = '''
Solving tasks and switching approaches:
If several attempts do not yield a result — change the approach.
Reformulate, simplify, or split the task,
work around limitations, look for an alternative path.
If there is no progress or actions repeat —
tell Milana about it.
Problematic if:
- no new information appears
- responses are not related to the task
- logic is lost
- tools return incorrect results.

Working with functions:
Check whether the result matches the request.
If a function returns incorrect, incomplete, disjointed, or meaningless output — do not repeat the same call unchanged.
Try changing the request, using another function, or doing without. Do not get stuck on one function'''
        return
    # Determine whether this is recreation or first creation
    if global_state.conversations % 2 == 0: # RECREATE executor
        return_text = main.create_executor_return_text_2
        let_log('ПЕРЕСОЗДАНИЕ ИСПОЛНИТЕЛЯ')
        lt = global_state.last_task_for_executor.get(global_state.conversations, '')
        # Check for exact task match
        if text == lt:
            return main.tasks_identical_text
        if text != '' and text is not None:
            param = parse_prompt_response(main.create_executor_param_1, main.create_executor_param_2 + ' 1:\n' + text + '\n' + main.create_executor_param_2 + ' 2:\n' + lt, 0)
        else:
            param = 0
        if param == 1: tag = 'correct'
        else: tag = 'incorrect'
        let_log(f"Пересоздание исполнителя, тег='{tag}' (param={param}, задача {'та же' if param == 0 else 'разная'})")
        save_emb_dialog(tag, 'executor')
        let_log(f"Сохранили старого исполнителя с тегом '{tag}'")
        delete_chat(global_state.conversations)
        let_log("Удалили старый чат исполнителя")
    else: # FIRST CREATION of executor at this level
        return_text = main.create_executor_return_text_1
        global_state.conversations += 1
        let_log('создание нового специалиста')
    global_state.last_task_for_executor[global_state.conversations] = text
    next_executor()
    # Get additional information through the librarian
    questions_raw = ask_model(text, system_prompt=gigo_questions)
    additional_info = librarian(questions_raw)
    if additional_info != found_info_1:
        additional_info = main.additional_info_text + additional_info
    else:
        additional_info = ''
        let_log("Библиотекарь не нашел дополнительной информации")
    ivan_tools = global_state.ivan_module_tools.copy()
    
    # Determine whether this executor can delegate
    current_level = get_level()
    if global_state.hierarchy_limit == 0:
        delegation_allowed = True
    else:
        delegation_allowed = current_level < global_state.hierarchy_limit
    # Remove delegation command if not available
    if global_state.hierarchy_limit == 1 and global_state.start_dialog_command_name in ivan_tools:
        del ivan_tools[global_state.start_dialog_command_name]
        let_log("Удалена команда делегирования из инструментов исполнителя")
    
    # Tool selection for the executor
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
    # Build string with tool descriptions for the prompt, excluding skip commands
    selected_ivan_tools = ''
    for tool in ivan_tools:
        if tool not in global_state.skip_tools_keys:
            selected_ivan_tools += tool + ' (' + ivan_tools[tool][0] + ')\n'
    # Determine which prompt to use (with or without tools)
    if selected_ivan_tools:
        system_prompt_for_instructions = main.create_executor_write_prompt_1
        user_content = text + additional_info + f"\n\n{main.avaiable_tools_text}\n" + selected_ivan_tools
    else:
        system_prompt_for_instructions = main.create_executor_write_prompt_2
        user_content = text + additional_info
    # Generate instructions for the executor based on the task
    let_log(system_prompt_for_instructions)
    let_log("Генерация инструкций для исполнителя...")
    instructions = ask_model(
        user_content,
        system_prompt=system_prompt_for_instructions
    )
    # Build final prompt for the executor
    prompt = main.worker_base + no_markdown_instruction + write_shortly_prompt + '\n' + prompt_evaluation_2 + ' ' + text
    if global_state.hierarchy_limit != 1:
        prompt += main.worker_delegation_part

    # Add hierarchy information if limit is greater than 1
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

    # Add example command for requesting information (always if not native call)
    if not native_func_call:
        prompt += what_is_func_text + main.need_info_example
    prompt += main.exec_anti_loop_text
    # Save tools for this chat
    global_state.tools_commands_dict[global_state.conversations] = ivan_tools
    let_log('ДОСТУПНЫЕ ИНСТРУМЕНТЫ ДЛЯ ИСПОЛНИТЕЛЯ:')
    for tool, (desc, _) in ivan_tools.items(): let_log(f"  {tool}: {desc}")
    # Create chat for the executor via manager
    system_prompt = system_role_text + prompt
    create_chat(global_state.conversations, system_prompt)
    # Return only the creation fact, without greeting hint
    return return_text