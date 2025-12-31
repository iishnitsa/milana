# optimize where text is similar
# better to save all this to a database, even text from external modules
# 1 database check 2 tokenization from here and from modules and loaders 3 writing to global variables in system code (glob or OOP in modules, not sure)

select_little_best_text = 'Select the best solution option that matches the task. If the first option is better, send 1, if the second, send 2, if they are the same, send 0.\nTask:\n'

solution_text_1 = '\nSolution 1:\n'

solution_text_2 = '\nSolution 2:\n'

is_similar_text = 'Are these solutions similar enough to be considered identical? If yes, send 1, if no, send 0.\nTask:\n'

summarize_prompt = '''
Summarize the text, briefly retelling while preserving important details. 
Please provide a brief overview, highlighting the most significant points and ensuring the original meaning and context are preserved: 
'''

found_info_1 = "Nothing found"

slash_n = '\n'

grouping_prompt_1 = """Analyze the following dialogue and group messages that are logically related.
For example: if several messages in a row pertain to the same question and answer, they can be combined.

"""

grouping_prompt_2 = """

Return only the message numbers that should be combined, in the format: 1,2-4,5,6-8
Only numbers, commas, and dashes. No explanations.

Examples:
- "1,2-5,6" (message 1 separately, 2-5 together, 6 separately)
- "1-3,4-7" (1-3 together, 4-7 together)
- "1,2,3" (all messages separately)

Groups:"""

slash_token = '/'

dot_zpt_token = ';'

librarier_nothing_found = 'Nothing found'

kv_token = '"'

prompt_n = ' Prompt:\n'

voskl_zn_token = '!!!'

tool_selector_return_1 = '\nFunction: '

tool_selector_return_2 = 'Wrong command'

start_dialog_tool_text = 'Read the prompt for the operator and write the names of tools needed by this operator to control executor results, separated by commas. For example: "tool_name1, tool_name2". If none fit, write: "None". Available tools:\n'

# this can be combined

system_role_text = 'System: '

operator_role_text = '\nMilana: '

worker_role_text = '\nIvan: '

func_role_text = '\nFunction: '

wrong_command = 'Wrong command'

start_dialog_history = 'Brief dialogue history: '

make_exec_first = 'Create an executor before starting the dialogue'

udp_exec_if_needed = '\nUpdate executor if needed'

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
gigo_make_plan = 'Come up with a task completion plan based on the following data:\n'
gigo_return_1 = 'Task:\n'
gigo_return_2 = 'Plan:\n'
gigo_reaction = 'Thoughts of different people about the task:'

start_load_attachments_text = 'Attachments are being downloaded, which may take a long time...'
end_load_attachments_text = 'Attachments uploaded'

marker_decision_approve = "VERDICT: APPROVE"
marker_decision_revise = "VERDICT: REVISE"
marker_decision_unsure = "VERDICT: UNSURE"
marker_new_task = "NEW TASK:"
prompt_decomposition_1 = """
Analyze the following task and break it down into key, specific, and verifiable completion criteria as a numbered list. Your goal is to create a checklist for result evaluation.

Task:"""
prompt_decomposition_2 = """
Example output:
1. Criterion one.
2. Criterion two.

Output only the numbered list of criteria.
"""
prompt_evaluation_1 = "You are given the original task, the result of its execution, and a list of criteria for evaluation. Evaluate how well the result meets EACH criterion."
prompt_evaluation_2 = "Task:"
prompt_evaluation_3 = "Result:"
prompt_evaluation_4 = "Criteria:"
prompt_evaluation_5 = """
For each line from the criteria list, output a verdict. Use the markers [DONE] or [NOT DONE] at the beginning of each line, then provide a brief and clear explanation for your decision.

Example output:
[DONE] 1. Criterion one. Matches, because...
[NOT DONE] 2. Criterion two. Not fulfilled, because it lacks...
"""
prompt_decision_1 = "You are a senior system analyst. Make a final decision regarding the AI executor's work. You are provided with the original task, the result, and a detailed evaluation report."
prompt_decision_2 = "Original task:"
prompt_decision_3 = "Previous result:"
prompt_decision_4 = "Evaluation report:"
prompt_decision_5 = """
Analyze all the data and deliver a verdict.

Your answer must have a STRICT structure:
First, on a separate line, your verdict. This can be one of three options: `VERDICT: APPROVE`, `VERDICT: REVISE`, or `VERDICT: UNSURE`.

- Use `VERDICT: REVISE` only if you see clear errors and can formulate a task to fix them.
- Use `VERDICT: UNSURE` if the result seems acceptable, but you cannot guarantee its completeness or correctness, or if you don't know how it can be improved.

If the verdict is `VERDICT: REVISE`, then AFTER it, starting with the marker `NEW TASK:`, formulate a COMPLETE, SELF-CONTAINED TASK for another AI executor.

Example output for revision:
VERDICT: REVISE
NEW TASK:
Your previous attempt to solve the task "write a summation function" was almost successful, but it lacked handling of non-numeric data. Please revise this function by adding a try-except block.
"""
prompt_librarian_questions_1 = """
You are a meticulous fact-checker. Based on the task, the result, and the evaluation report, formulate a list of questions to ask an external knowledge source ("the librarian") in order to verify facts, find best practices, or identify hidden errors.

Task:"""
prompt_librarian_questions_2 = "Result:"
prompt_librarian_questions_3 = "Evaluation report:"
prompt_librarian_questions_4 = """
If there are no questions, return an empty string. If there are, output only the questions themselves, each on a new line. Ask only those questions whose answers will genuinely help improve the result.
"""
prompt_decision_librarian_context = "Additional context from the librarian:\n"

summarize_text_some_phrases = '''
Summarize the following text into an annotation of 5-10 sentences. Return only the finished annotation, without any introductory words, explanations, or concluding phrases. Start immediately with the first sentence of the annotation.
Text:\n'''

annotation_available_prompt = 'Annotation of available information:\n'

err_image_process_text_infoloaders = 'Error processing image: '
text_on_image_prompt_infoloaders = '\nText on image:'
err_image_process_pdf_infoloaders = 'Error opening PDF: '
page_pdf_prompt_infoloaders = 'Page'
attachment_prompt_infoloaders = 'Attachment '
unprocessable_file_infoloaders = "Unprocessable file"
file_processing_error_infoloaders = "File processing error"
image_processing_error_infoloaders = "Image processing error"
corrupted_zip_infoloaders = "Error: corrupted ZIP archive"
zip_processing_error_infoloaders = "ZIP archive processing error"
unsupported_format_infoloaders = "Format not supported for processing"
file_open_error_infoloaders = "Error opening file: "
zip_archive_name_infoloaders = "ZIP archive"
model_early_loading_error_text = "Loading of image processing models previously ended with error" # import
excel_cheet_text = "Sheet: "
excel_cheet_size_text = "Size: "
excel_cheet_strings_text = "rows,"
excel_cheet_columns_text = "columns"
excel_cheet_data_text = "Data:"
excel_cheet_error_text = "Error processing sheet"
excel_empty_text = "Excel file contains no data or an error occurred while reading"
excel_error_text = "Error processing Excel file:"
annotation_failed_text = "Failed to create annotation for uploaded files"

user_review_text1 = 'Task:\n'
user_review_text2 = '\nResult:\n'
user_review_text3 = '\nClient reaction:\n'
user_review_text4 = '\nAI critic reaction:\n'

what_is_func_text = '''

If the interlocutor starts their message with the text "Function: ", then it is not the interlocutor, but a system message
or a response from a function if it was called by you.
The interlocutor does not know that you are calling a function and does not see the function's response.

'''

last_messages_marker = "\nLast messages:"

rag_context_marker = "\nContext (previous messages from long-term memory):\n"

global_summary_marker = "\nGlobal dialogue summary:\n"

recent_summary_marker = "\nSummary of the last topic:\n"

text_tokens_coefficient = 0.5 # middle coefficient for english

class SystemTextContainer:
    def __init__(self):
        self.select_little_best_text = select_little_best_text
        self.solution_text_1 = solution_text_1
        self.solution_text_2 = solution_text_2
        self.is_similar_text = is_similar_text
        self.summarize_prompt = summarize_prompt
        self.found_info_1 = found_info_1
        self.slash_n = slash_n
        self.grouping_prompt_1 = grouping_prompt_1
        self.grouping_prompt_2 = grouping_prompt_2
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
        self.make_exec_first = make_exec_first
        self.udp_exec_if_needed = udp_exec_if_needed
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
        self.marker_decision_approve = marker_decision_approve
        self.marker_decision_revise = marker_decision_revise
        self.marker_decision_unsure = marker_decision_unsure
        self.marker_new_task = marker_new_task
        self.prompt_decomposition_1 = prompt_decomposition_1
        self.prompt_decomposition_2 = prompt_decomposition_2
        self.prompt_evaluation_1 = prompt_evaluation_1
        self.prompt_evaluation_2 = prompt_evaluation_2
        self.prompt_evaluation_3 = prompt_evaluation_3
        self.prompt_evaluation_4 = prompt_evaluation_4
        self.prompt_evaluation_5 = prompt_evaluation_5
        self.prompt_decision_1 = prompt_decision_1
        self.prompt_decision_2 = prompt_decision_2
        self.prompt_decision_3 = prompt_decision_3
        self.prompt_decision_4 = prompt_decision_4
        self.prompt_decision_5 = prompt_decision_5
        self.prompt_librarian_questions_1 = prompt_librarian_questions_1
        self.prompt_librarian_questions_2 = prompt_librarian_questions_2
        self.prompt_librarian_questions_3 = prompt_librarian_questions_3
        self.prompt_librarian_questions_4 = prompt_librarian_questions_4
        self.prompt_decision_librarian_context = prompt_decision_librarian_context
        self.summarize_text_some_phrases = summarize_text_some_phrases
        self.annotation_available_prompt = annotation_available_prompt
        self.err_image_process_text_infoloaders = err_image_process_text_infoloaders
        self.text_on_image_prompt_infoloaders = text_on_image_prompt_infoloaders
        self.err_image_process_pdf_infoloaders = err_image_process_pdf_infoloaders
        self.page_pdf_prompt_infoloaders = page_pdf_prompt_infoloaders
        self.attachment_prompt_infoloaders = attachment_prompt_infoloaders
        self.unprocessable_file_infoloaders = unprocessable_file_infoloaders
        self.file_processing_error_infoloaders = file_processing_error_infoloaders
        self.image_processing_error_infoloaders = image_processing_error_infoloaders
        self.corrupted_zip_infoloaders = corrupted_zip_infoloaders
        self.zip_processing_error_infoloaders = zip_processing_error_infoloaders
        self.unsupported_format_infoloaders = unsupported_format_infoloaders
        self.file_open_error_infoloaders = file_open_error_infoloaders
        self.zip_archive_name_infoloaders = zip_archive_name_infoloaders
        self.model_early_loading_error_text = model_early_loading_error_text
        self.excel_cheet_text = excel_cheet_text
        self.excel_cheet_size_text = excel_cheet_size_text
        self.excel_cheet_strings_text = excel_cheet_strings_text
        self.excel_cheet_columns_text = excel_cheet_columns_text
        self.excel_cheet_data_text = excel_cheet_data_text
        self.excel_cheet_error_text = excel_cheet_error_text
        self.excel_empty_text = excel_empty_text
        self.excel_error_text = excel_error_text
        self.annotation_failed_text = annotation_failed_text
        self.user_review_text1 = user_review_text1
        self.user_review_text2 = user_review_text2
        self.user_review_text3 = user_review_text3
        self.user_review_text4 = user_review_text4
        self.what_is_func_text = what_is_func_text
        self.last_messages_marker = last_messages_marker
        self.rag_context_marker = rag_context_marker
        self.global_summary_marker = global_summary_marker
        self.recent_summary_marker = recent_summary_marker
        self.text_tokens_coefficient = text_tokens_coefficient

def system_text_container():
    return SystemTextContainer()