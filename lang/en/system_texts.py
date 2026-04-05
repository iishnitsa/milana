summarize_prompt = '''
Summarize the text, briefly retelling while preserving important details.
Give a brief overview, highlighting the most significant points and preserving the original meaning and context.
Do not shorten ```highlighted text```, quotes, links, formulas, code, or other elements that must not be shortened.
Return only the finished summary, without any introductory words, explanations, or concluding phrases. Start immediately with the first sentence of the summary.
'''

found_info_1 = "Nothing found"

grouping_prompt_1 = """Analyze the dialogue and group messages that are logically related.
For example: if several messages in a row refer to the same question and answer, they can be combined.
"""

grouping_prompt_2 = """
Return only the message numbers that need to be combined, in the format: 1,2-4,5,6-8
Only numbers, commas, and dashes. No explanations.
Examples:
- "1,2-5,6" (message 1 separately, 2-5 together, 6 separately)
- "1-3,4-7" (1-3 together, 4-7 together)
- "1,2,3" (all messages separately)
Groups:"""

# this can be combined

system_role_text = 'System: '

operator_role_text = '\nMilana: '

worker_role_text = '\nIvan: '

func_role_text = '\nFunction: '

start_dialog_history = 'Brief dialogue history: '

make_exec_first = 'Create an executor before starting the dialogue'

gigo_dreamer = 'dreamer'
gigo_realist = 'realist'
gigo_critic = 'critic'
gigo_questions = 'The user will send you a task. In response, write a question or questions about what information is missing to complete the task. One question per line. If there are several questions, ask all at once in one message. Each question must be self-contained, since the system does not take context into account when searching. For example, "What authoritative sources study phenomenon X?" instead of "What authoritative sources are there?". Send only questions, no other text should be present'
gigo_found_info = 'Only the following information is available:'
gigo_dreamer_note = '. Your task is to propose the boldest, most ambitious and ideal solution, not limited by resources or current capabilities. Imagine how to complete the task in the best possible way to leave the client completely delighted'
gigo_realist_note = '. Your task is to propose a practical, feasible solution. Describe how to effectively complete the task, avoiding unnecessary complications'
gigo_critic_note = '. Your task is to analyze possible solutions and point out their weaknesses, risks, and potential problems. Identify what could go wrong and suggest how to avoid or mitigate the consequences'
gigo_role_answer_1 = 'You are a '
gigo_role_answer_2 = '. The user will send you a task and, possibly, additional information to complete the task. Respond immediately with how to perfectly complete the task, how to fully satisfy the client whose task the user sent. You cannot ask the user questions or discuss anything with them. An answer is needed immediately. The answer should not contain interrogative sentences.'
gigo_make_plan_1 = '''The user will send you the thoughts of different entities about solving the given task.
As soon as they send the thoughts of the last entity, immediately write a plan for solving the task with a length of 10-25 lines.
Do not comment on it, do not write anything like "Here is a plan for solving...", do not ask questions.
The answer should not contain interrogative sentences.
Just the plan.
'''
gigo_make_plan_2 = '\nEntities: '
gigo_return_1 = 'Task:\n'
gigo_return_2 = 'Plan:\n'
gigo_next_role = 'Next: '
gigo_final_role = ". That's it! Right after your message I will send the plan."
gigo_final_role_2 = "This is the last one. I am waiting for a plan from you right now!"

start_load_attachments_text = 'Attachments are being loaded, this may take a long time...'
end_load_attachments_text = 'Attachments loaded'

marker_decision_approve = "VERDICT: APPROVE"
marker_decision_revise = "VERDICT: REVISE"
marker_decision_unsure = "VERDICT: UNSURE"
marker_new_task = "NEW TASK:"
prompt_decomposition_1 = """
Analyze the following task and break it down into key, specific, and verifiable completion criteria as a numbered list. Your goal is to create a checklist for evaluating the result.
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
For each line from the criteria list, output a verdict. Use the markers [DONE] or [NOT DONE] at the beginning of each line, then give a brief and clear explanation of your decision.
Example output:
[DONE] 1. Criterion one. Matches, because...
[NOT DONE] 2. Criterion two. Not fulfilled, because it lacks...
"""
prompt_decision_1 = "You are a senior system analyst. Make a final decision on the work of the AI executor. You are provided with the original task, the result, and a detailed evaluation report. Note: the result may be a message about the impossibility of the task with evidence. Your task is to check whether the task is really impossible or whether it is an executor error."
prompt_decision_2 = "Original task:"
prompt_decision_3 = "Previous result:"
prompt_decision_4 = "Evaluation report:"
prompt_decision_5 = """
Analyze all the data and deliver a verdict.
Your answer must have a STRICT structure:
First, on a separate line, your verdict. This can be one of three options: `VERDICT: APPROVE`, `VERDICT: REVISE`, or `VERDICT: UNSURE`.
- Use `VERDICT: REVISE` only if you see clear errors and can formulate a task to fix them.
- Use `VERDICT: UNSURE` if the result seems acceptable, but you cannot guarantee its completeness or correctness, if you do not know how it can be improved, OR if after analyzing the evidence you conclude that the task is objectively impossible (see criteria below).

If the verdict is `VERDICT: REVISE`, then AFTER it, starting with the marker `NEW TASK:`, formulate a COMPLETE, SELF-CONTAINED TASK for another AI executor.

**Criteria for impossibility (a situation is considered problematic/absurd if):**
- actions repeat without progress
- responses lose connection to the task
- the direction of reasoning constantly shifts
- tools return inconsistent, contradictory, or useless results
- environment or tool limitations make the goal unreachable
- required data or functions are missing or unavailable
- all reasonable approaches lead to repetition or absurdity

If the result contains a justification of impossibility:
- Check it against the criteria above.
- If it meets the criteria → verdict `VERDICT: UNSURE`.
- If it does not (e.g., the executor gives up without valid reasons) → verdict `VERDICT: REVISE` with a new task stating that the previous claim of impossibility was incorrect and describing how to proceed.

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
model_early_loading_error_text = "Loading of image processing models previously ended with error"
excel_cheet_text = "Sheet: "
excel_cheet_size_text = "Size: "
excel_cheet_strings_text = "rows,"
excel_cheet_columns_text = "columns"
excel_cheet_data_text = "Data:"
excel_cheet_error_text = "Error processing sheet"
excel_empty_text = "Excel file contains no data or an error occurred while reading"
excel_error_text = "Error processing Excel file:"
annotation_failed_text = "Failed to create annotation for uploaded files"

user_review_text2 = '\nResult:\n'
user_review_text3 = '\nClient reaction:\n'
user_review_text4 = '\nAI critic reaction:\n'

what_is_func_text = '''
To call a command, write three exclamation marks at the beginning of the message, then the command name, then three more exclamation marks, and then the information for the command.
Do not use json or markdown to call functions.
If you are writing a command, write only the command, do not comment on your actions.
If the interlocutor starts their message with the text "Function: ", then it is not the interlocutor, but a system message or a function response if it was called by you.
'''

only_one_func_text = """
Only one command is allowed per message. A command cannot be combined with a message for the interlocutor.
The interlocutor will not receive a message containing a correctly written and system-recognized command, nor will they see the function's response to your command.
At the same time, for the interlocutor to receive the message, do not write a command!
You can only call commands available in the system instruction. In parentheses is a description of their functions:
"""

last_messages_marker = "\nLast messages:"

rag_context_marker = """
Retrieved fragments from long-term memory.
They were found by meaning and may contain important details from earlier parts of the dialogue that are not included in the recent messages.
Use them as an additional source of facts and context, but keep in mind that they may be incomplete, erroneous, or not sorted by time.
Use them to refine your answer, but do not contradict the explicit history of recent messages.
"""

global_summary_marker = "\nGlobal dialogue summary (overall picture, key decisions and facts from the entire conversation):\n"

recent_summary_marker = "\nSummary of the recent topic (brief summary of the recent part of the dialogue not yet included in the global summary):\n"

error_in_provider = 'An error occurred while accessing the model provider. Infinite retry attempts are being made with a 60-second interval. You can stop the program if a successful request message does not appear for a long time'

success_in_provider = 'The error has been resolved, continuing work'

wrong_command = 'Wrong command'

warn_command_text_1 = "Protocol violation detected:"

warn_command_text_2 = "Only one command per message is allowed."

warn_command_text_3 = "Command must start at the beginning of the message."

warn_command_text_4 = "Command is inside a Markdown block. Call commands outside formatting."

warn_command_text_5 = "Command is inside a JSON structure. Use pure !!!command!!! format."

warn_command_text_6 = 'If you did NOT try to invoke a command, use !!!skip!!! at the very beginning of the message, then write your message again — it will be sent to the interlocutor (for example, "!!!skip!!! I want to say that...").'

warn_command_text_7 = "Command is inside markdown formatting (bold, italic, code, etc.)."

no_markdown_instruction = 'IMPORTANT: Do not use Markdown (e.g., **bold**, *italic*, `code`, lists with * or -) in your responses. Write in plain text. Markdown is allowed only if explicitly required for formatted code or data, but in regular conversation avoid it.'

yes_no_instruction = """
You must answer only "Yes." or "No." Do not add any additional text, explanations, or punctuation other than a single period at the end. This is critically important for the system to parse your response correctly.
Examples of correct answers:
- Yes.
- No.
Examples of incorrect answers:
- Yes, I think so.
- No, because...
- Probably yes.
- Yep.
- Definitely not.
Always answer exactly "Yes." or "No." depending on your decision.
"""

yes_word = "Yes."

no_word = "No."

cut_message_prompt = '''You are an AI message compressor. The user will send you their message.
Your task is to reply with that same message on behalf of the user, paraphrasing to shorten it, without giving any comments. Just remove the fluff.
If several words can be replaced by one, do so. No detail should be lost.
Do not shorten ```highlighted text```, !!!commands!!!, quotes, links, formulas, code, or other elements that must not be shortened.
If there is nothing to shorten, simply rewrite the message unchanged.
The user's message will be replaced by your message and embedded into the conversation, which is why you cannot give comments, lead-ins, or any extraneous text.'''

write_shortly_prompt = '\nTry to write concisely to save context.'

prompt_chunk_summary = "Concisely summarize the essence of the given dialogue fragment. Highlight key facts, decisions, and important details. Response — 1-2 sentences."

prompt_global_summary = "Based on the summaries of individual fragments, create a single global summary of the entire dialogue. Describe the main topics, decisions made, and key facts. Use 2-4 sentences."

prompt_recent_summary = "Based on the summaries of fragments, create a brief summary of the recent conversation topic. Highlight the essence of the discussion and important details. Use 1-3 sentences."

text_tokens_coefficient = 0.5 # average coefficient for English

class SystemTextContainer:
    def __init__(self):
        self.summarize_prompt = summarize_prompt
        self.found_info_1 = found_info_1
        self.grouping_prompt_1 = grouping_prompt_1
        self.grouping_prompt_2 = grouping_prompt_2
        self.system_role_text = system_role_text
        self.operator_role_text = operator_role_text
        self.worker_role_text = worker_role_text
        self.func_role_text = func_role_text
        self.start_dialog_history = start_dialog_history
        self.make_exec_first = make_exec_first
        self.gigo_dreamer = gigo_dreamer
        self.gigo_realist = gigo_realist
        self.gigo_critic = gigo_critic
        self.gigo_questions = gigo_questions
        self.gigo_found_info = gigo_found_info
        self.gigo_dreamer_note = gigo_dreamer_note
        self.gigo_realist_note = gigo_realist_note
        self.gigo_critic_note = gigo_critic_note
        self.gigo_role_answer_1 = gigo_role_answer_1
        self.gigo_role_answer_2 = gigo_role_answer_2
        self.gigo_make_plan_1 = gigo_make_plan_1
        self.gigo_make_plan_2 = gigo_make_plan_2
        self.gigo_return_1 = gigo_return_1
        self.gigo_return_2 = gigo_return_2
        self.gigo_next_role = gigo_next_role
        self.gigo_final_role = gigo_final_role
        self.gigo_final_role_2 = gigo_final_role_2
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
        self.user_review_text2 = user_review_text2
        self.user_review_text3 = user_review_text3
        self.user_review_text4 = user_review_text4
        self.what_is_func_text = what_is_func_text
        self.only_one_func_text = only_one_func_text
        self.last_messages_marker = last_messages_marker
        self.rag_context_marker = rag_context_marker
        self.global_summary_marker = global_summary_marker
        self.recent_summary_marker = recent_summary_marker
        self.error_in_provider = error_in_provider
        self.success_in_provider = success_in_provider
        self.wrong_command = wrong_command
        self.warn_command_text_1 = warn_command_text_1
        self.warn_command_text_2 = warn_command_text_2
        self.warn_command_text_3 = warn_command_text_3
        self.warn_command_text_4 = warn_command_text_4
        self.warn_command_text_5 = warn_command_text_5
        self.warn_command_text_6 = warn_command_text_6
        self.warn_command_text_7 = warn_command_text_7
        self.no_markdown_instruction = no_markdown_instruction
        self.yes_no_instruction = yes_no_instruction
        self.yes_word = yes_word
        self.no_word = no_word
        self.cut_message_prompt = cut_message_prompt
        self.write_shortly_prompt = write_shortly_prompt
        self.prompt_chunk_summary = prompt_chunk_summary
        self.prompt_global_summary = prompt_global_summary
        self.prompt_recent_summary = prompt_recent_summary
        self.text_tokens_coefficient = text_tokens_coefficient

def system_text_container(): return SystemTextContainer()