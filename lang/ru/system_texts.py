select_little_best_text = 'Выбери лучший вариант решения, соответствующий задаче, если первый вариант лучше, отправь 1, если второй, отправь 2, если они одинаковы, отправь 0.\nЗадача:\n'

solution_text_1 = '\nРешение 1:\n'

solution_text_2 = '\nРешение 2:\n'

is_similar_text = 'Достаточно ли похожи данные решения, чтобы назвать их одинаковыми? Если да, отправь 1, если нет, отправь 0.\nЗадача:\n'

summarize_prompt = '''
Резюмируй текст, кратко пересказывая и сохраняя важные детали. 
Пожалуйста, дай краткий обзор, выделив наиболее значимые моменты и обеспечив сохранение первоначального смысла и контекста: 
'''

found_info_1 = "Не удалось ничего найти"

grouping_prompt_1 = """Проанализируй следующий диалог и сгруппируй сообщения, которые логически связаны.
Например: если несколько сообщений подряд относятся к одному вопросу и ответу, их можно объединить.

"""
grouping_prompt_2 = """

Верни только номера сообщений, которые нужно объединить, в формате: 1,2-4,5,6-8
Только цифры, запятые и тире. Без пояснений.

Примеры:
- "1,2-5,6" (сообщения 1 отдельно, 2-5 вместе, 6 отдельно)
- "1-3,4-7" (1-3 вместе, 4-7 вместе)
- "1,2,3" (все сообщения отдельно)

Группы:"""

# это можно объединить

system_role_text = 'Система: '

operator_role_text = '\nМилана: '

worker_role_text = '\nИван: '

func_role_text = '\nФункция: '

start_dialog_history = 'Краткая история диалога: '

make_exec_first = 'Создай исполнителя перед началом диалога'

udp_exec_if_needed = '\nОбнови исполнителя, если нужно'

gigo_dreamer = 'мечтатель'
gigo_realist = 'реалист'
gigo_critic = 'критик'
gigo_questions = 'Пользователь пришлёт тебе задачу. В ответ напиши вопрос или вопросы, какой информации нехватает для выполнения задачи. По одному вопросу на строку. Задавай сразу все вопросы в одном сообщении, если вопросов несколько. Пришли только вопросы, никакого другого другого текста быть не должно'
gigo_found_info = 'Доступна только следующая информация:'
gigo_dreamer_note = '. Твоя задача — предложить самое смелое, амбициозное и идеальное решение, не ограниченное ресурсами или текущими возможностями. Представь, как можно выполнить задачу наилучшим образом, чтобы клиент был в полном восторге'
gigo_realist_note = '. Твоя задача — предложить практичное, выполнимое решение. Опиши, как эффективно выполнить задачу, избегая излишних сложностей'
gigo_critic_note = '. Твоя задача — проанализировать возможные решения и указать на их слабые места, риски, потенциальные проблемы. Выяви, что может пойти не так, и предложи, как этого избежать или смягчить последствия'
gigo_role_answer_1 = 'Ты - '
gigo_role_answer_2 = '. Пользователь пришлёт тебе задачу и, возможно, дополнительную информацию для выполнения задачи. Ответь сразу, как идеально выполнить задачу, как сполна удовлетворить клиента, задачу которого прислал пользователь. Нельзя задавать вопросов пользователю, или обсуждать с ним что-то. Нужен сразу ответ. Ответ не должен содержать вопросительных предложений.'
gigo_make_plan_1 = '''Пользователь будет присылать тебе мысли разных сущностей о решении поставленной задачи.
Как только он пришлёт мысли последней сущности, сразу напиши план решения задачи.
Не комментируй его, не пиши что-то вроде "Вот план по решению...", не задавай вопросов.
Ответ не должен содержать вопросительных предложений.
Просто план.

'''
gigo_make_plan_2 = '\nСущности: '
gigo_return_1 = 'Задача:\n'
gigo_return_2 = 'План:\n'
gigo_next_role = 'Далее: '
gigo_final_role = '. Всё! Сразу после твоего сообщения я пришлю план.'
gigo_final_role_2 = 'Это последнее. Жду от тебя план прямо сейчас!'
start_load_attachments_text = 'Происходит загрузка вложений, может занять много времени...'
end_load_attachments_text = 'Вложения загружены'

marker_decision_approve = "ВЕРДИКТ: ПРИНЯТЬ"
marker_decision_revise = "ВЕРДИКТ: ДОРАБОТАТЬ"
marker_decision_unsure = "ВЕРДИКТ: НЕ УВЕРЕН"
marker_new_task = "НОВАЯ ЗАДАЧА:"
prompt_decomposition_1 = """
Проанализируй следующую задачу и разбей её на ключевые, конкретные и проверяемые критерии выполнения в виде нумерованного списка. Твоя цель — создать чек-лист для оценки результата.

Задача:"""
prompt_decomposition_2 = """
Пример вывода:
1. Критерий один.
2. Критерий два.

Выведи только нумерованный список критериев.
"""
prompt_evaluation_1 = "Тебе дана исходная задача, результат её выполнения и список критерий для оценки. Оцени, насколько результат соответствует КАЖДОМУ критерию."
prompt_evaluation_2 = "Задача:"
prompt_evaluation_3 = "Результат:"
prompt_evaluation_4 = "Критерии:"
prompt_evaluation_5 = """
Для каждой строки из списка критериев выведи вердикт. Используй маркеры [ВЫПОЛНЕНО] или [НЕ ВЫПОЛНЕНО] в начале каждой строки, а затем дай краткое и чёткое объяснение твоего решения.

Пример вывода:
[ВЫПОЛНЕНО] 1. Критерий один. Соответствует, так как...
[НЕ ВЫПОЛНЕНО] 2. Критерий два. Не выполнен, потому что отсутствует...
"""
prompt_decision_1 = "Ты — старший системный аналитик. Прими финальное решение по работе AI-исполнителя. Тебе предоставлена исходная задача, результат и детальный отчёт о проверке."
prompt_decision_2 = "Исходная задача:"
prompt_decision_3 = "Предыдущий результат:"
prompt_decision_4 = "Отчёт об оценке:"
prompt_decision_5 = """
Проанализируй все данные и вынеси вердикт.

Твой ответ должен иметь СТРОГУЮ структуру:
Сначала на отдельной строке твой вердикт. Это может быть один из трёх вариантов: `ВЕРДИКТ: ПРИНЯТЬ`, `ВЕРДИКТ: ДОРАБОТАТЬ` или ``.

- Используй `ВЕРДИКТ: ДОРАБОТАТЬ`, только если ты видишь чёткие ошибки и можешь сформулировать задачу по их исправлению.
- Используй `ВЕРДИКТ: НЕ УВЕРЕН`, если результат выглядит приемлемо, но ты не можешь гарантировать его полноту или корректность, или если ты не знаешь, как его можно улучшить.

Если вердикт `ВЕРДИКТ: ДОРАБОТАТЬ`, то ПОСЛЕ него, начиная с маркера `НОВАЯ ЗАДАЧА:`, сформулируй ПОЛНОЦЕННОЕ, САМОДОСТАТОЧНОЕ ЗАДАНИЕ для другого AI-исполнителя.

Пример вывода для доработки:
ВЕРДИКТ: ДОРАБОТАТЬ
НОВАЯ ЗАДАЧА:
Твоя предыдущая попытка решить задачу "написать функцию суммирования" была почти успешной, но в ней отсутствовала обработка нечисловых данных. Пожалуйста, доработай эту функцию, добавив блок try-except.
"""
prompt_librarian_questions_1 = """
Ты — дотошный факт-чекер. Основываясь на задаче, результате и отчёте об оценке, сформулируй список вопросов, которые нужно задать внешнему источнику знаний ("библиотекарю"), чтобы проверить факты, найти лучшие практики или выявить скрытые ошибки.

Задача:"""
prompt_librarian_questions_2 = "Результат:"
prompt_librarian_questions_3 = "Отчёт об оценке:"
prompt_librarian_questions_4 = """
Если вопросов нет, верни пустую строку. Если есть, выведи только сами вопросы, каждый на новой строке. Задавай только те вопросы, ответы на которые действительно помогут улучшить результат.
"""
prompt_decision_librarian_context = "Дополнительный контекст от библиотекаря:\n"

summarize_text_some_phrases = '''
Сократи следующий текст до аннотации объемом 5-10 предложений. Верни только готовую аннотацию, без любых вступительных слов, пояснений или заключительных фраз. Начни сразу с первого предложения аннотации.
Текст:\n'''

annotation_available_prompt = 'Аннотация доступной информации:\n'

err_image_process_text_infoloaders = 'Ошибка при обработке изображения: '
text_on_image_prompt_infoloaders = '\nТекст на изображении:'
err_image_process_pdf_infoloaders = 'Ошибка открытия PDF: '
page_pdf_prompt_infoloaders = 'Страница'
attachment_prompt_infoloaders = 'Attachment '
unprocessable_file_infoloaders = "Необрабатываемый файл"
file_processing_error_infoloaders = "Ошибка обработки файла"
image_processing_error_infoloaders = "Ошибка обработки изображения"
corrupted_zip_infoloaders = "Ошибка: поврежденный ZIP-архив"
zip_processing_error_infoloaders = "Ошибка обработки ZIP-архива"
unsupported_format_infoloaders = "Формат не поддерживается для обработки"
file_open_error_infoloaders = "Ошибка открытия файла: "
zip_archive_name_infoloaders = "ZIP-архив"
model_early_loading_error_text = "Загрузка моделей обработки изображений ранее завершилась ошибкой" #импорт
excel_cheet_text = "Лист: "
excel_cheet_size_text = "Размер: "
excel_cheet_strings_text = "строк,"
excel_cheet_columns_text = "столбцов"
excel_cheet_data_text = "Данные:"
excel_cheet_error_text = "Ошибка при обработке листа"
excel_empty_text = "Файл Excel не содержит данных или произошла ошибка при чтении"
excel_error_text = "Ошибка при обработке Excel файла:"
annotation_failed_text = "Не удалось создать аннотацию для загруженных файлов"

user_review_text2 = '\nРезультат:\n'
user_review_text3 = '\nРеакция клиента:\n'
user_review_text4 = '\nРеакция ИИ-критика:\n'

what_is_func_text = '''
Чтобы вызвать команду, напиши в начале сообщения три восклицательных знака, потом имя команды, потом ещё три восклицательных знака, а потом информацию для команды.
Не используй json и markdown для вызова функций.
Если ты пишешь команду, пиши только команду, не комментируй свои действия.
Если собеседник начинает своё сообщение текстом "Функция: ", то это не собеседник, а системное сообщение
или ответ функции, если она была тобой вызвана.

'''

only_one_func_text = """
В одном сообщении разрешено писать одну команду. Команда не может быть совмещена с сообщением для собеседника.
Собеседник не получит сообщение содержащее правильно написанную и распознанную системой.
Собеседник не увидит ответ функции на твою команду.
Ты можешь вызывать только команды, доступные в системной инструкции. В скобках у них описание их функций:

"""

last_messages_marker = "\nПоследние сообщения:"

rag_context_marker = "\nКонтекст (предыдущие сообщения из долгосрочной памяти):\n"

global_summary_marker = "\nГлобальная сводка диалога:\n"

recent_summary_marker = "\nСводка последней темы:\n"

error_in_provider = 'Произошла ошибка при обращении к провайдеру модели. Производятся бесконечные попытки запроса с интервалом в 60 секунд. Вы можете остановить работу программы, если сообщение об успешном запросе долго не появляется'

success_in_provider = 'Ошибка исчезла, продолжаю работу'

wrong_command = 'Неправильная команда'

warn_command_text_1 = "Обнаружено нарушение протокола:"

warn_command_text_2 = "Разрешена только одна команда в сообщении."

warn_command_text_3 = "Команда должна начинаться с начала сообщения."

warn_command_text_4 = "Команда находится внутри Markdown-блока. Вызывайте команды вне форматирования."

warn_command_text_5 = "Команда находится внутри JSON-структуры. Используйте чистый формат !!!команда!!!."

warn_command_text_6 = "Если вы НЕ пытались вызвать команду, используйте !!!пропустить!!! и затем напишите ваше сообщение ещё раз — оно будет отправлено собеседнику."

text_tokens_coefficient = 0.5 # усреднённый коэффициент для русского языка

class SystemTextContainer:
    def __init__(self):
        self.select_little_best_text = select_little_best_text
        self.solution_text_1 = solution_text_1
        self.solution_text_2 = solution_text_2
        self.is_similar_text = is_similar_text
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
        self.udp_exec_if_needed = udp_exec_if_needed
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
        self.text_tokens_coefficient = text_tokens_coefficient

def system_text_container(): return SystemTextContainer()