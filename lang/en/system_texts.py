# optimize where text is similar
# better to save all this to a database, even text from external modules
# 1 database check 2 tokenization from here and from modules and loaders 3 writing to global variables in system code (glob or OOP in modules, not sure)

select_little_best_text = 'Select the best solution option that matches the task. If the first option is better, send 1, if the second, send 2, if they are the same, send 0.\nTask:\n'

solution_text_1 = '\nSolution 1:\n'

solution_text_2 = '\nSolution 2:\n'

is_similar_text = 'Are these solutions similar enough to be considered identical? If yes, send 1, if no, send 0.\nTask:\n'

milana_template = '''
You are curator Milana.
 The client sends you a task plan.
 You create specialist Ivan for the task by sending the task to the tool.
 Then you sequentially outline and send each subtask to your executor Ivan, creating a specialist for each task.
 If you're dissatisfied with the result - inform Ivan and detail the reason for dissatisfaction.
 When you're confident that the overall task is completed - end the dialogue.
 Client's task: 

'''

summarize_prompt = '''
Summarize the text, briefly retelling while preserving important details. 
Please provide a brief overview, highlighting the most significant points and ensuring the original meaning and context are preserved: 
'''

found_info_1 = "Nothing found"

save_emb_dialog_history = 'Brief dialogue history'

slash_n = '\n'

save_emb_dialog_mark_thesis_1 = 'Extract theses from text:\n'

save_emb_dialog_mark_thesis_2 = '\nMark each thesis with a line !!!thesis!!!'

save_emb_dialog_thesis = '!!!thesis!!!'

save_emb_dialog_mark_group = 'Divide the dialogue into meaningful groups. Between groups put a line !!!group!!!.\n'

save_emb_dialog_group = '!!!group!!!'

slash_token = '/'

dot_zpt_token = ';'

librarier_nothing_found = 'Nothing found'

kv_token = '"'

prompt_n = ' Prompt:\n'

voskl_zn_token = '!!!'

tool_selector_return_1 = '\nFunction: '

tool_selector_return_2 = 'Wrong command'

start_dialog_tool_text = 'Read the prompt for the operator and write the names of tools needed by this operator to control specialist results, separated by commas. For example: "tool_name1, tool_name2". If none fit, write: "None". Available tools:\n'

# this can be combined

system_role_text = 'System: '

operator_role_text = '\nMilana: '

worker_role_text = '\nIvan: '

func_role_text = '\nFunction: '

wrong_command = 'Wrong command'

start_dialog_history = 'Brief dialogue history: '

make_spec_first = 'Create a specialist before starting the dialogue'

udp_spec_if_needed = '\nUpdate specialist if needed'

tst = ' - '

dot_space = '. '

just_space = ' '

zpt_space = ', '

zpt_token = ','

space_skb = ' ('

closing_skb = ')'

q_token = '?'

null_token = '0'

one_token = '1'

two_token = '2'

three_token = '3'

gigo_dreamer = 'dreamer'
gigo_realist = 'realist'
gigo_critic = 'critic'
gigo_questions = 'Write a question or questions, one per line, what information is missing to complete the task: '
gigo_found_info = 'Only the following information is available, there will be no other information, and you can not request more or ask questions either: '
gigo_not_found_info = "There is no more information about the task and there won't be, and you can't request information or ask questions either."
gigo_role_answer_1 = 'You are a '
gigo_role_answer_2 = '. Tell me how to fully satisfy the client who set the task: '
gigo_make_plan = 'Come up with a task completion plan based on the following data:\nThe task: '
gigo_return_1 = 'Task: '
gigo_return_2 = 'Plan: '
gigo_reaction = 'Thoughts of different people about the task:'

start_load_attachments_text = 'Attachments are being downloaded, which may take a long time...'
end_load_attachments_text = 'Attachments uploaded'

class SystemTextContainer:
    def __init__(self):
        self.select_little_best_text = select_little_best_text
        self.solution_text_1 = solution_text_1
        self.solution_text_2 = solution_text_2
        self.is_similar_text = is_similar_text
        self.milana_template = milana_template
        self.summarize_prompt = summarize_prompt
        self.found_info_1 = found_info_1
        self.save_emb_dialog_history = save_emb_dialog_history
        self.slash_n = slash_n
        self.save_emb_dialog_mark_thesis_1 = save_emb_dialog_mark_thesis_1
        self.save_emb_dialog_mark_thesis_2 = save_emb_dialog_mark_thesis_2
        self.save_emb_dialog_thesis = save_emb_dialog_thesis
        self.save_emb_dialog_mark_group = save_emb_dialog_mark_group
        self.save_emb_dialog_group = save_emb_dialog_group
        self.slash_token = slash_token
        self.dot_zpt_token = dot_zpt_token
        self.librarier_nothing_found = librarier_nothing_found
        self.kv_token = kv_token
        self.prompt_n = prompt_n
        self.voskl_zn_token = voskl_zn_token
        self.tool_selector_return_1 = tool_selector_return_1
        self.tool_selector_return_2 = tool_selector_return_2
        self.start_dialog_tool_text = start_dialog_tool_text
        self.system_role_text = system_role_text
        self.operator_role_text = operator_role_text
        self.worker_role_text = worker_role_text
        self.func_role_text = func_role_text
        self.wrong_command = wrong_command
        self.start_dialog_history = start_dialog_history
        self.make_spec_first = make_spec_first
        self.udp_spec_if_needed = udp_spec_if_needed
        self.tst = tst
        self.dot_space = dot_space
        self.just_space = just_space
        self.zpt_space = zpt_space
        self.zpt_token = zpt_token
        self.space_skb = space_skb
        self.closing_skb = closing_skb
        self.q_token = q_token
        self.null_token = null_token
        self.one_token = one_token
        self.two_token = two_token
        self.three_token = three_token
        self.gigo_dreamer = gigo_dreamer
        self.gigo_realist = gigo_realist
        self.gigo_critic = gigo_critic
        self.gigo_questions = gigo_questions
        self.gigo_found_info = gigo_found_info
        self.gigo_not_found_info = gigo_not_found_info
        self.gigo_role_answer_1 = gigo_role_answer_1
        self.gigo_role_answer_2 = gigo_role_answer_2
        self.gigo_make_plan = gigo_make_plan
        self.gigo_return_1 = gigo_return_1
        self.gigo_return_2 = gigo_return_2
        self.gigo_reaction = gigo_reaction
        self.start_load_attachments_text = start_load_attachments_text
        self.end_load_attachments_text = end_load_attachments_text

def system_text_container():
    return SystemTextContainer()