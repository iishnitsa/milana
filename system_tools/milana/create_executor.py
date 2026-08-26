'''
create_executor
receives a task, creates an executor and, possibly, provides them with tools suitable for solving this task
'''

from cross_gpt import (
    found_info_1,
    find_all_commands,
    only_one_func_text,
    what_is_func_text,
    what_is_func_text_not_at_start,
    allow_command_not_at_start,
    give_operator_goal_to_executor,
    native_func_call,
    let_log,
    global_state,
    ask_model,
    text_cutter,
    sql_exec,
    save_emb_dialog,
    librarian,
    librarian_search_surface_ok,
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
    use_magical_prompt,
    use_psm,
    give_all_tools,
    use_librarian,
    cacher,
)

@cacher
def _executor_tools_pick_and_instructions(
    task_text,
    additional_info,
    tools_str,
    module_keys,
    give_all,
    catalog,
    select_1,
    select_2,
    write_1,
    write_2,
    available_label,
    base_selected_lines,
    skip_keys,
):
    """
    One cache slot: module tool selection + instruction text for executor.
    Returns (selected_module_names: list, instructions: str, selected_ivan_tools_text: str).
    Nested ask_model uses its own @cacher slots.
    """
    module_keys = list(module_keys or ())
    skip_keys = set(skip_keys or ())
    catalog = list(catalog or ())
    names = []
    if module_keys:
        if give_all:
            names = list(module_keys)
            let_log('[give_all_tools] все модули отданы исполнителю (cached path)')
        else:
            need_tools_raw = ask_model(
                task_text,
                system_prompt=select_1 + tools_str + select_2,
            )
            let_log('Результат выбора инструментов:')
            let_log(need_tools_raw)
            names = find_all_commands(need_tools_raw, module_keys)
            let_log(f"Найдены инструменты: {names}")
    catalog_map = {n: d for n, d in catalog}
    extra = ''
    for n in names:
        if n in skip_keys:
            continue
        if n in catalog_map:
            extra += f"{n} ({catalog_map[n]})\n"
            let_log(f"Добавлен инструмент: {n}")
    selected_text = (base_selected_lines or '') + extra
    if selected_text.strip():
        sys_p = write_1
        user_c = task_text + additional_info + f"\n\n{available_label}\n" + selected_text
    else:
        sys_p = write_2
        user_c = task_text + additional_info
    let_log(sys_p)
    let_log("Генерация инструкций для исполнителя...")
    instructions = ask_model(user_c, system_prompt=sys_p)
    return (list(names), instructions if instructions is not None else '', selected_text)


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
            'need_info_example_note',
            'need_info_example_unavailable',
            'operator_goal_label',
            'tasks_identical_text',
            'exec_anti_loop_text',
            'exec_magical',
            'exec_psm_prompt_1',
            'exec_psm_prompt_2',)
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
Example command call - "!!!internal_search!!! React hooks documentation"
'''
        main.need_info_example_note = '''
This is only an example of command syntax.
'''
        main.need_info_example_unavailable = '''
Currently this particular command is not available.
'''
        main.operator_goal_label = (
            '\nOperator goal (context only; this is the operator\'s objective without the plan, '
            'not a replacement for your subtask — follow the task and instructions above):\n'
        )
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
        main.exec_magical = """
Execution principles:

Never claim to have performed an action unless it has actually been performed.
Never fabricate results, files, observations, or tool outputs.
If you are uncertain about something, clearly distinguish assumptions from verified facts.

If one approach fails, deliberately try a different one instead of repeating the same actions.

Consider not only the intended use of available tools, but also unconventional uses, provided they comply with the system's rules and may help accomplish the task.

If completing the task requires an action that only Milana or the user can perform, report it honestly and prepare everything necessary for the work to continue.

If the task cannot be completed in full, try to complete the largest possible part of it or produce the most useful intermediate result.

If several genuinely different approaches still produce no meaningful progress, explain to Milana what prevents further progress instead of pretending that the task has been completed.
"""
        main.exec_psm_prompt_1 = """
Below are the user's task and the operator's personality.

Describe, in a single short sentence, the personality of an executor who would form an effective team with the operator.

The executor should not be a copy of the operator.

Their personality and thinking style should differ enough that they notice what the other might overlook, naturally encourage each other toward higher-quality work, and explore the task from different perspectives without creating constant conflict.

For example:
- caution may complement experimentation;
- thorough analysis may complement rapid hypothesis testing;
- strategic thinking may complement practicality;
- creativity may complement critical thinking;
- skepticism may complement initiative.

Describe only:
- personality;
- thinking style;
- internal motivation.

Do not describe skills, professions, or knowledge.
Do not mention the task itself.
Do not use names, famous people, or fictional characters.
Do not describe appearance, age, biography, or speaking style.

Return only one sentence of no more than 30 words.

The task:
"""
        main.exec_psm_prompt_2 = """
Personality:
"""
        return
    if global_state.conversations % 2 == 0:
        return_text = main.create_executor_return_text_2
        let_log('ПЕРЕСОЗДАНИЕ ИСПОЛНИТЕЛЯ')
        max_rec = int(getattr(global_state, 'max_executor_recreates', 0) or 0)
        cur_rec = int(getattr(global_state, 'executor_recreate_count', 0) or 0)
        if max_rec > 0 and cur_rec >= max_rec:
            let_log(f"[create_executor] лимит пересозданий: {cur_rec}/{max_rec}")
            return f"Executor recreate limit reached ({max_rec}). Continue with the current executor or end the dialog."
        lt = global_state.last_task_for_executor.get(global_state.conversations, '')
        if text == lt: return main.tasks_identical_text
        if text != '' and text is not None: param = parse_prompt_response(main.create_executor_param_1, main.create_executor_param_2 + ' 1:\n' + text + '\n' + main.create_executor_param_2 + ' 2:\n' + lt, 0)
        else: param = 0
        if param == 1: tag = 'correct'
        else: tag = 'incorrect'
        let_log(f"Пересоздание исполнителя, тег='{tag}' (param={param}, задача {'та же' if param == 0 else 'разная'})")
        save_emb_dialog(tag, 'executor')
        let_log(f"Сохранили старого исполнителя с тегом '{tag}'")
        delete_chat(global_state.conversations)
        let_log("Удалили старый чат исполнителя")
        global_state.executor_recreate_count = cur_rec + 1
    else: return_text = main.create_executor_return_text_1; global_state.conversations += 1; global_state.executor_recreate_count = 0; let_log('создание нового специалиста')
    global_state.last_task_for_executor[global_state.conversations] = text
    next_executor()
    additional_info = ''
    if use_librarian and librarian_search_surface_ok():
        questions_raw = ask_model(text, system_prompt=gigo_questions)
        additional_info = librarian(questions_raw)
        if additional_info != found_info_1:
            additional_info = main.additional_info_text + additional_info
        else:
            additional_info = ''
            let_log("Библиотекарь не нашел дополнительной информации")
    elif use_librarian:
        let_log("Библиотекарь: нет surface (web/chroma) — вопросы не генерируем")
    else:
        let_log("Библиотекарь выключен (use_librarian=0)")
    ivan_tools = global_state.ivan_module_tools.copy()
    current_level = get_level()
    if global_state.hierarchy_limit == 0: delegation_allowed = True
    else: delegation_allowed = current_level < global_state.hierarchy_limit
    if global_state.hierarchy_limit == 1 and global_state.start_dialog_command_name in ivan_tools: del ivan_tools[global_state.start_dialog_command_name]; let_log("Удалена команда делегирования из инструментов исполнителя")
    skip = tuple(global_state.skip_tools_keys or [])
    base_selected = ''
    for tool in ivan_tools:
        if tool not in skip:
            base_selected += tool + ' (' + ivan_tools[tool][0] + ')\n'
    catalog = tuple(
        (tool_tokens, tool_desc)
        for tool_tokens, tool_desc, _tool_func in (global_state.another_tools or [])
    )
    tools_names, instructions, selected_ivan_tools = _executor_tools_pick_and_instructions(
        text or '',
        additional_info or '',
        global_state.tools_str or '',
        tuple(global_state.module_tools_keys or []),
        bool(give_all_tools),
        catalog,
        main.create_executor_select_tools_1,
        main.create_executor_select_tools_2,
        main.create_executor_write_prompt_1,
        main.create_executor_write_prompt_2,
        main.avaiable_tools_text,
        base_selected,
        skip,
    )
    for name in tools_names or []:
        for tool_tokens, tool_desc, tool_func in (global_state.another_tools or []):
            if name == tool_tokens:
                ivan_tools[tool_tokens] = (tool_desc, tool_func)
                break
    prompt = main.worker_base
    if use_psm:
        from cross_gpt import psm_get
        oper_person = psm_get(global_state.conversations - 1, 'per', '') or ''
        prompt += ' ' + ask_model(main.exec_psm_prompt_1 + text + main.exec_psm_prompt_2 + oper_person, all_user=True)
    prompt += no_markdown_instruction + write_shortly_prompt + '\n' + prompt_evaluation_2 + ' ' + text
    if global_state.hierarchy_limit != 1: prompt += main.worker_delegation_part
    hierarchy_note = ""
    if global_state.hierarchy_limit > 1:
        limit = global_state.hierarchy_limit
        hierarchy_note = f"\n{main.hierarchy_limit_info} {current_level}/{limit}.\n"
        if not delegation_allowed: hierarchy_note += f"\n{main.delegate_unavailable_for_executor}\n"
    prompt += hierarchy_note
    prompt += instructions + only_one_func_text
    # явный список доступных команд с описаниями
    if selected_ivan_tools:
        prompt += '\n' + main.avaiable_tools_text + '\n' + selected_ivan_tools
    if not native_func_call:
        # template for calling commands (always); concrete examples optional (tools_no_examples)
        if allow_command_not_at_start and what_is_func_text_not_at_start:
            prompt += what_is_func_text_not_at_start
        else:
            prompt += what_is_func_text
        no_examples = bool(getattr(global_state, 'tools_no_examples', False))
        need_info_available = any(
            any(tag in str(k).lower() for tag in (
                'need_info', 'нужна_информац', 'internal_search', 'внутренний_поиск', 'librarian'))
            for k in (ivan_tools or {})
            if k not in global_state.skip_tools_keys
        )
        if not no_examples:
            prompt += main.need_info_example
            # need_info (librarian): если команды нет — пример + «недоступна»; если есть — только пример
            if not need_info_available and not use_librarian:
                prompt += getattr(main, 'need_info_example_note', '') or '\nThis is only an example of command syntax.\n'
                prompt += getattr(main, 'need_info_example_unavailable', '') or (
                    '\nCurrently this particular command is not available.\n')
        elif not need_info_available and not use_librarian:
            prompt += getattr(main, 'need_info_example_unavailable', '') or (
                '\nCurrently internal_search / librarian is not available.\n')
    prompt += main.exec_anti_loop_text
    if use_magical_prompt: prompt += main.exec_magical
    # Optional: operator goal without GIGO plan (context only; concrete task is still create_executor arg)
    if give_operator_goal_to_executor:
        raw_goal = getattr(global_state, 'main_now_task', '') or ''
        goal = raw_goal
        for marker in ('\nPlan:\n', '\nПлан:\n', '\nplan:\n', '\nплан:\n'):
            if marker in goal:
                goal = goal.split(marker, 1)[0]
                break
        # strip leading "Task:" / "Задача:" labels if present
        for prefix in ('Task:\n', 'Задача:\n', 'Task:', 'Задача:'):
            if goal.lstrip().startswith(prefix):
                goal = goal.lstrip()[len(prefix):]
                break
        goal = goal.strip()
        if goal:
            label = getattr(main, 'operator_goal_label', None) or (
                '\nOperator goal (context only; this is the operator\'s objective, not your full brief — follow the subtask above):\n')
            prompt += label + goal + '\n'
    from cross_gpt import set_agent_tools
    set_agent_tools(global_state.conversations, ivan_tools, role='executor')
    let_log('ДОСТУПНЫЕ ИНСТРУМЕНТЫ ДЛЯ ИСПОЛНИТЕЛЯ:')
    for tool, (desc, _) in ivan_tools.items(): let_log(f"  {tool}: {desc}")
    system_prompt = system_role_text + prompt
    create_chat(global_state.conversations, system_prompt)
    return return_text