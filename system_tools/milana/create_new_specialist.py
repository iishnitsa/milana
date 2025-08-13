'''
new_specialist
receives a task and finds a specialist for it
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
    let_log
)

def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'create_new_specialist_param_1',
            'create_new_specialist_param_2',
            'create_new_specialist_questions',
            'additional_info_text',
            'create_new_specialist_write_prompt_1',
            #'create_new_specialist_write_prompt_2',
            'create_new_specialist_select_tools_1',
            'create_new_specialist_select_tools_2',
            'worker_tool_prompt',
            'create_new_specialist_return_text',
        )
        main.create_new_specialist_param_1 = 'Are the tasks identical?\nTask 1:'
        main.create_new_specialist_param_2 = '\nIf they are different, send "1", if identical - "0"'
        main.create_new_specialist_questions = 'Write questions, separating them with ; to search for additional information for this task: '
        main.additional_info_text = 'Additional information: '
        main.create_new_specialist_write_prompt_1 = 'Write a prompt for the described specialist named Ivan, or one needed to complete the described task (don\'t forget to include the task in the prompt):\n'
        #main.create_new_specialist_write_prompt_2 = '\nUse this prompt as a base and expand it: '
        main.create_new_specialist_select_tools_1 = '''You're reading a task description for a specialist. Based on the task description, select tools from the list of allowed ones.

Output data:

Only tool names separated by comma and space in a single line.
No quotes, no periods at the end, no explanations and no additional characters.
If no tools are suitable or if there's any incorrect or extra name, output exactly: None

List of allowed tools:

'''
        main.create_new_specialist_select_tools_2 = '''

RESTRICTIONS:

It's forbidden to output any other tool names except those from the list.
You cannot invent new commands.
Any typo or incorrect entry in the list is sufficient reason to output None.

Task description:

'''
        main.worker_tool_prompt = '''

Your supervisor is Milana. You perform any task assigned by Milana. 
Don't rush to answer immediately, perform the task sequentially, discussing each step with Milana, listening to her comments. 
If the task is too complex, or Milana is dissatisfied with the result after several attempts, 
initiate creating a new dialog, describing the reason for dissatisfaction if any, and then pass the result to Milana. 

\nTo call a tool (only one call per response), you need to start your response with three exclamation marks, then the tool name, then three more exclamation marks, and then the information for the tool. The command must be sent as is, without interpretation, for example: "!!!new_dialog!!!Calculate mathematical model".\nYou have access to the following tools: '

'''
        main.create_new_specialist_return_text = 'Specialist created, start working with them'
        return

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
        chunk_size,
    )
    import difflib
    set_now_try()

    if global_state.conversations % 2 == 0: # ОН ДОЛЖЕН УДАЛИТЬ ЧАТ С ПРОШЛЫМ СПЕЦИАЛИСТОМ
        let_log('ПЕРЕСОЗДАНИЕ специалиста')
        lt = global_state.last_task_for_specialist[global_state.conversations]
        while True:
            try:
                param = int(ask_model(
                    main.create_new_specialist_param_1 + 
                    text + 
                    main.create_new_specialist_param_2
                ))
                break
            except:
                continue

        parts = global_state.get_now_try().split(slash_token) # может тут из-за токена ошибка была?
        parents_id = slash_token.join(parts[:-1]) if len(parts) > 1 else slash_token

        sql_exec(
            'INSERT INTO tries (try_id, task, parents_id) VALUES (?, ?, ?)',
            (parts[-1], global_state.main_now_task, parents_id)
        )

        history = sql_exec(
            'SELECT history FROM chats WHERE chat_id=?',
            (global_state.conversations,), fetchone=True
        )
        save_emb_dialog(history, 'incorrect' if param == 0 else 'correct') # TODO:
        sql_exec("DELETE FROM chats WHERE chat_id = ?", (global_state.conversations,))

    else:
        global_state.conversations += 1

    additional_info = ''
    questions_raw = ask_model(main.create_new_specialist_questions + text)
    questions = questions_raw.split(zpt_space)

    for q in questions:
        la = librarian(q)
        if la != found_info_1:
            additional_info += la + slash_n
    if additional_info != '':
        #while len(additional_info) > chunk_size:
        #    text_cutter(additional_info)
        additional_info = (
            main.additional_info_text +
            additional_info
        )

    prompt = ask_model(
        main.create_new_specialist_write_prompt_1 +
        text +
        additional_info
    )

    #let_log(prompt)

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
            main.create_new_specialist_select_tools_1 +
            global_state.tools_str +
            main.create_new_specialist_select_tools_2 +
            prompt
        )
        need_tools_raw = ask_model(prompt_tools)

        # Новый участок — замена старой логики
        let_log('ВОТ')
        let_log(need_tools_raw)
        tools_names = [t.strip().replace('\\_','_') for t in need_tools_raw.split(',') if t.strip()]
        tools_names = list(dict.fromkeys(tools_names))  # Удаляем дубликаты

        for name in tools_names:
            found = False
            for tool_tokens, tool_desc, tool_func in global_state.another_tools:
                tool_str = tool_tokens
                if name in tool_str:
                    ivan_tools[tool_tokens] = tool_desc, tool_func
                    found = True
                    break
            if not found:
                #possible = difflib.get_close_matches(name, [t for t in global_state.another_tools.keys()], n=1, cutoff=0.75)
                first_lists = [t[0] for t in global_state.another_tools]
                possible = difflib.get_close_matches(name, [item for sublist in first_lists for item in sublist], n=1, cutoff=0.75)
                if possible:
                    for tool_tokens, tool_desc, tool_func in global_state.another_tools:
                        if tool_tokens == possible[0]:
                            ivan_tools[tool_tokens] = tool_desc, tool_func
                            break
    if ivan_tools: prompt += main.worker_tool_prompt
    for tool in ivan_tools:
        prompt += tool + space_skb + ivan_tools[tool][0] + closing_skb + zpt_token
    
    if prompt and prompt[-1] == zpt_token:
        prompt = prompt[:-1]

    global_state.tools_commands_dict[global_state.conversations] = ivan_tools

    let_log('ДОСТУПНЫЕ ИНСТРУМЕНТЫ:')
    let_log(global_state.tools_commands_dict)
    
    sql_exec(
        'INSERT INTO chats (chat_id, prompt, history) VALUES (?, ?, ?)',
        (global_state.conversations, prompt, '')
    )

    global_state.stop_agent = True
    return main.create_new_specialist_return_text