'''
delegate_task
transfers the task for division into subtasks and execution
'''

from cross_gpt import (
    make_exec_first,
    system_role_text,
    operator_role_text,
    func_role_text,
    wrong_command,
    start_dialog_history,
    remove_commands_roles,
    find_all_commands,
    only_one_func_text,
    what_is_func_text,
    what_is_func_text_not_at_start,
    allow_command_not_at_start,
    last_messages_marker,
    native_func_call,
    let_log,
    global_state,
    ask_model,
    text_cutter,
    gigo,
    gigo_adv,
    use_old_gigo,
    give_all_tools,
    module_hints_for_operator,
    tools_selector,
    create_executor,
    down_hierarchy,
    get_level,
    save_emb_dialog,
    create_chat,
    update_history,
    get_chat_context,
    no_markdown_instruction,
    write_shortly_prompt,
    use_magical_prompt,
    use_psm,)

def find_tuple_by_first_list(data, target_list):
    for list1, list2, obj in data:
        if list1 == target_list: return (list1, list2), obj
    return None

def _format_tools_for_prompt(tools_dict):
    """Описания tools в промпт (без skip), module-level — не пересоздаётся в main."""
    lines = []
    skip = set(global_state.skip_tools_keys or [])
    for tool, meta in (tools_dict or {}).items():
        if tool in skip:
            continue
        desc = meta[0] if isinstance(meta, (tuple, list)) and meta else str(meta)
        lines.append(f"{tool} ({desc})")
    return ('\n'.join(lines) + '\n') if lines else ''

def main(client_task):
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'start_dialog_tool_text_1',
            'start_dialog_tool_text_2',
            'milana_base_1',
            'milana_base_2',
            'milana_base_3',
            'milana_delegation_part',
            'hierarchy_limit_info',
            'oper_anti_loop_text',
            'delegate_unavailable_for_operator',
            'conversations_limit_reached_text',
            'oper_magical',
            'oper_psm_prompt',
        )
        main.milana_base_1 = '''You are "Milana", an AI operator. Above you is the CLIENT (human) — they send the original task and receive the final result. You have received a task plan from the client'''
        main.milana_base_2 = ''' or from a HIGHER-LEVEL AGENT (another operator/dialog above you in the hierarchy). That higher agent may send a refined plan and task instead of the human client. Treat their request as the current authority for this dialog'''
        main.milana_base_3 = '''
Your workflow:
1. CREATE ONE EXECUTOR — Use the command "!!!create_executor!!!" followed by the task description. This creates "Ivan", an AI executor who will handle the current subtask.
CORRECT: "!!!create_executor!!! *current subtask*"
INCORRECT: "!!!create_executor!!! Create Ivan for *current subtask*"
INCORRECT: "!!!create_executor!!! Ivan, *current subtask*"
2. WORK WITH THE SAME EXECUTOR — After creation, continue the conversation with Ivan. Give instructions, answer questions, receive results. DO NOT create a new executor unless absolutely necessary.
IMPORTANT: Once Ivan is created, NEVER use the "!!!create_executor!!!" command again during your conversation with him. This command is only for the initial creation. Using it again will cause errors and unnecessarily recreate the executor.
3. WHEN RECREATION IS ALLOWED — Recreate Ivan ONLY in these cases:
- The current executor explicitly states they CANNOT complete the task.
- The task changes so drastically that a different specialization is needed.
- Ivan SUCCESSFULLY completed their part and you need a new executor for a clearly separate next subtask (but first try to have Ivan handle multiple steps).
4. FORBIDDEN — Never recreate an executor IMMEDIATELY after they respond. A simple reply from Ivan is NOT a reason to create a new one. Continue the conversation.
5. GREETING AFTER CREATION — After successfully creating Ivan, simply greet him in natural language (e.g., "Hello, Ivan."). Do not include any commands in your greeting.
IMPORTANT ABOUT THE EXECUTOR: Ivan does NOT see the full client task/plan you received — only the subtask text you put after create_executor. Make that subtask self-contained; do not assume he knows the original client request or the full plan.
6. DELEGATION — '''
        main.milana_delegation_part = '''Ivan can delegate a subtask further down the hierarchy if he cannot handle it. This creates a similar dialog which you (Milana) cannot access. Only Ivan has the right to delegate.
'''
        main.start_dialog_tool_text_1 = '''You will receive a plan and a task from the user. Based on the task description, select tools from the list of allowed tools.
Output data:
Only tool names separated by comma and space in a single line.
No quotes, no periods at the end, no explanations, and no additional characters.
If no tool is suitable or if at least one incorrect or extra name is found, output exactly: None
List of allowed tools:
'''
        main.start_dialog_tool_text_2 = '''
LIMITATIONS:
It is forbidden to output any other tool names except those from the list.
You cannot invent new commands.
Any typo or incorrect entry in the list is sufficient reason to output None.
'''
        main.oper_anti_loop_text = '''
Execution control and dialogue termination:
Monitor the executor’s progress. If there is no solution, determine whether
the issue is task complexity or impossibility.
Consider the situation problematic (absurd) if:
- actions repeat without progress
- results lose connection to the task
- the direction of reasoning constantly shifts
- tools return inconsistent, contradictory, or useless results
Do not stop the dialogue due to complexity alone.
First try adjusting the plan or decomposition.
Mark the task as impossible only if:
- environment or tool limitations make the goal unreachable
- required data or functions are missing or unavailable
- all reasonable approaches lead to repetition or absurdity
When stopping, you must prove impossibility:
- list attempts and why they failed
- describe constraints or failures
- explain why further attempts will not succeed'''
        main.hierarchy_limit_info = 'Hierarchy levels are limited. Current level'
        main.delegate_unavailable_for_operator = 'The task delegation function down the hierarchy is not available to you.'
        main.conversations_limit_reached_text = 'The limit of delegation levels has been reached. The task has not been transferred.'
        main.oper_magical = """
Execution principles:

Never report an action as completed unless it has actually been completed.
Never present assumptions as facts.
If you are uncertain about something, state it explicitly.

Your goal is to solve the client's problem, not to finish the conversation as quickly as possible.
If the task is completed poorly, incompletely, or with a false claim of success, the client will most likely submit it again. Therefore, pretending success provides no benefit compared to making real progress.

Before concluding that a task is impossible, explore the space of possible solutions.

Consider:
- whether the plan can be changed;
- whether the task can be decomposed differently;
- whether a different part of the work can be delegated to the executor;
- whether other available tools can be used;
- whether several tools can be combined;
- whether a useful intermediate result can be produced;
- whether materials can be prepared for the user;
- whether the user can be asked to perform an action unavailable to the system so that the work can continue afterward.

Do not stop working simply because one step cannot currently be completed. If you can still provide value in another way, do so.

Conclude that a task is impossible only after multiple reasonable approaches have been explored and you can explain why further attempts are unlikely to succeed.
"""
        main.oper_psm_prompt = """
Below is the user's task.

Write the personality as a role line starting with "You are..." (or "Ты..." if the target language is Russian).

Describe, in a single short sentence, the personality of an operator who would be best suited for solving this task.

Describe only:
- personality;
- thinking style;
- internal motivation.

Do not describe skills, professions, or knowledge.
Do not mention the task itself.
Do not use names, famous people, or fictional characters.
Do not describe appearance, age, biography, or speaking style.

Create a natural and psychologically consistent personality that can later be effectively complemented by another team member.

The personality should help organize the work, make good decisions, and guide the task toward the most useful achievable outcome.

Return only one sentence of no more than 30 words.

The task:
"""
        return
    let_log('начинается диалог')
    if global_state.hierarchy_limit != 0 and global_state.hierarchy_limit == get_level(): return main.conversations_limit_reached_text
    global_state.dialog_ended = False
    if global_state.critic_wants_retry: global_state.critic_wants_retry = False
    else:
        if global_state.conversations > 0:
            let_log('СОХРАНЕНИЕ перед делегированием')
            save_emb_dialog('delegated')
            let_log('Сохранили оператора как delegated')
            if global_state.conversations % 2 == 0 and global_state.conversations != 0: save_emb_dialog('delegated', 'executor'); let_log('Сохранили исполнителя как delegated')
    global_state.stop_agent = True
    _gigo_fn = gigo if use_old_gigo else gigo_adv
    if client_task == '': prompt = _gigo_fn(global_state.main_now_task); global_state.retries = False
    else: global_state.main_now_task = client_task; prompt = _gigo_fn(client_task)
    if global_state.summ_attach != global_state.summ_attach: prompt += global_state.summ_attach; global_state.summ_attach = ''
    # === DELEGATION: add a new level ===
    down_hierarchy()
    current_level = get_level()
    let_log(f"После делегирования: {global_state.now_try}, текущий уровень: {current_level}")
    if global_state.hierarchy_limit == 0: ivan_can_delegate = True
    else: ivan_can_delegate = (current_level + 1) < global_state.hierarchy_limit
    # Tool selection for Milana
    milana_tools = global_state.milana_module_tools.copy()
    if global_state.module_tools_keys:
        if give_all_tools:
            # Опция: не выбирать подмножество — отдать все модули
            for tool_tokens, tool_desc, tool_func in global_state.another_tools:
                milana_tools[tool_tokens] = (tool_desc, tool_func)
            let_log('[give_all_tools] все модули отданы оператору')
        else:
            need_tools_raw = ask_model(prompt, system_prompt=main.start_dialog_tool_text_1 + global_state.tools_str + main.start_dialog_tool_text_2)
            let_log(need_tools_raw)
            tools_names = find_all_commands(need_tools_raw, global_state.module_tools_keys)
            # Remove delegation command from selected tools if it accidentally got in
            if not ivan_can_delegate and global_state.start_dialog_command_name in tools_names:
                tools_names.remove(global_state.start_dialog_command_name)
                let_log(f"Удалена команда делегирования из выбранных инструментов")
            for name in tools_names:
                for tool_tokens, tool_desc, tool_func in global_state.another_tools:
                    if name == tool_tokens: milana_tools[tool_tokens] = (tool_desc, tool_func); break
            let_log('ошибки нет')
    # Remove delegation command from Milana's tools if the next level is unavailable
    if not ivan_can_delegate and global_state.start_dialog_command_name in milana_tools:
        del milana_tools[global_state.start_dialog_command_name]
        let_log("Удалена команда делегирования из инструментов Миланы")
    # === BUILD PROMPT ===
    # client vs higher agent: level>1 → делегирование сверху, не «голый» client
    full_prompt = main.milana_base_1
    if current_level > 1 or global_state.hierarchy_limit != 1:
        full_prompt += main.milana_base_2
    # 1. Information that Ivan can delegate (added ALWAYS except when delegation is completely disabled - limit=1)
    if global_state.hierarchy_limit != 1:
        if use_psm:
            operator_personality = ask_model(main.oper_psm_prompt + client_task, all_user=True)
            from cross_gpt import psm_set
            psm_set(global_state.conversations + 1, per=operator_personality)
            full_prompt += ' ' + operator_personality
        full_prompt += main.milana_base_3
        full_prompt += main.milana_delegation_part
        full_prompt += f"\n{main.delegate_unavailable_for_operator}\n"
        # 2. Message for Milana: she cannot delegate (added ALWAYS except when delegation is completely disabled - limit=1)
    else: full_prompt += main.milana_base_3
    # 3. Add hierarchy information if limit is greater than 1
    if global_state.hierarchy_limit > 1: full_prompt += f"\n{main.hierarchy_limit_info} {current_level}/{global_state.hierarchy_limit}.\n"
    full_prompt += no_markdown_instruction + write_shortly_prompt
    # mid-dialog client notes when deliver_user_messages is on (plain text reply, no answer_client)
    if getattr(global_state, 'deliver_user_messages', False):
        try:
            from cross_gpt import client_messages_operator_note
            full_prompt += client_messages_operator_note
        except Exception:
            full_prompt += (
                '\nThe external client may send messages while you work. '
                'You get a special turn: reply in plain text or skip to return to Ivan.\n'
            )
    let_log(milana_tools)
    prompt += only_one_func_text
    prompt += _format_tools_for_prompt(milana_tools)
    if not native_func_call:
        if allow_command_not_at_start and what_is_func_text_not_at_start:
            prompt += what_is_func_text_not_at_start
        else:
            prompt += what_is_func_text
    # Опционально: расширенные подсказки по модулям и оператору
    if module_hints_for_operator and global_state.another_tools:
        hints = '\n'.join(
            f"- {tok}: {desc}" for tok, desc, _ in global_state.another_tools
            if tok not in global_state.skip_tools_keys)
        if hints:
            prompt += '\nModule hints:\n' + hints + '\n'
    full_prompt += prompt + main.oper_anti_loop_text
    if use_magical_prompt: full_prompt += main.oper_magical
    let_log(full_prompt)
    let_log(global_state.another_tools)
    let_log(milana_tools)
    global_state.conversations += 1
    let_log(f"[DBG_MILANA_CREATE] conversations={global_state.conversations}")
    create_chat(global_state.conversations, system_role_text + full_prompt)
    update_history(global_state.conversations, make_exec_first, func_role_text)
    from cross_gpt import set_agent_tools
    set_agent_tools(global_state.conversations, milana_tools, role='operator')
    # Generate initial response (agent dialogue → large model via purpose='agent')
    try:
        talk_prompt = ask_model(
            system_role_text +
            full_prompt +
            last_messages_marker +
            func_role_text +
            make_exec_first +
            operator_role_text,
            purpose='agent')
    except:
        try:
            talk_prompt = ask_model(
                system_role_text +
                text_cutter(full_prompt) +
                last_messages_marker +
                func_role_text +
                make_exec_first +
                operator_role_text,
                purpose='agent')
        except: raise
    update_history(global_state.conversations, talk_prompt, operator_role_text)
    let_log("НАЧАЛЬНЫЙ ОТВЕТ МИЛАНЫ:")
    let_log(talk_prompt)
    talk_prompt_for_tools = remove_commands_roles(talk_prompt)
    let_log(f"[DBG_BEFORE_CREATE_EXECUTOR] conversations={global_state.conversations}")
    milana_chat_id = global_state.conversations
    answer = tools_selector(talk_prompt_for_tools, global_state.conversations)
    global_state.now_agent_id = global_state.conversations
    if answer != wrong_command and global_state.dialog_state and answer != None: talk_prompt = answer
    elif global_state.dialog_state: let_log('СОЗДАНИЕ НОВОГО ИСПОЛНИТЕЛЯ...'); talk_prompt = create_executor(talk_prompt)
    else: return answer
    let_log("ОТВЕТ ПОСЛЕ ОБРАБОТКИ:")
    let_log(talk_prompt)
    let_log(f"[DBG_AFTER_CREATE_EXECUTOR] conversations={global_state.conversations}")
    last_talk_prompt = talk_prompt # This is the response from function/tool
    # Get history for second call
    let_log(f"[DBG_CONTEXT_LOAD] requested_chat={global_state.conversations}, milana_chat_id={milana_chat_id}")
    _, history_for_model = get_chat_context(global_state.conversations - 1)
    try:
        talk_prompt = ask_model(
            system_role_text +
            full_prompt +
            history_for_model +
            func_role_text +
            last_talk_prompt +
            operator_role_text,
            purpose='agent')
    except:
        history_for_model = start_dialog_history + text_cutter(history_for_model)
        try:
            talk_prompt = ask_model(
                system_role_text +
                full_prompt +
                last_messages_marker +
                history_for_model +
                func_role_text +
                last_talk_prompt +
                operator_role_text,
                purpose='agent')
        except Exception as e:
            try:
                talk_prompt = ask_model(
                    system_role_text +
                    full_prompt +
                    last_messages_marker +
                    history_for_model +
                    func_role_text +
                    text_cutter(last_talk_prompt) +
                    operator_role_text,
                    purpose='agent')
            except: raise
    let_log(f"[DBG_HISTORY_WRITE] target_chat={global_state.conversations - 1}, current_conv={global_state.conversations}")
    update_history(global_state.conversations - 1, last_talk_prompt, func_role_text)
    update_history(global_state.conversations - 1, talk_prompt, operator_role_text, local_message=False)
    return talk_prompt