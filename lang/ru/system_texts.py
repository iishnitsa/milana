# оптимизируй там где текст похожий
# лучше это в базу данных сохранить всё, даже текст из внешних модулей
# 1 проверка бд 2 токенизация отсюда и из модулей и загрузчиков 3 запись в глобальные переменные в системном коде (в модулях глоб или ооп, хз)

select_little_best_text = 'Выбери лучший вариант решения, соответствующий задаче, если первый вариант лучше, отправь 1, если второй, отправь 2, если они одинаковы, отправь 0.\nЗадача:\n'

solution_text_1 = '\nРешение 1:\n'

solution_text_2 = '\nРешение 2:\n'

is_similar_text = 'Достаточно ли похожи данные решения, чтобы назвать их одинаковыми? Если да, отправь 1, если нет, отправь 0.\nЗадача:\n'

milana_template = '''
Ты - куратор Милана.
 Клиент присылает тебе план задач.
 Ты создаёшь специалиста Ивана для задачи, отправив задачу в инструмент.
 Затем по очереди расписываешь и присылаешь каждую подзадачу своему исполнителю Ивану, создавая специалиста для каждой задачи.
 Если ты недовольна результатом - сообщаешь Ивану и подробно расписываешь причину недовольства.
 Когда будешь уверена в том, что общая задача выполнена - завершаешь диалог.
 Задача от клиента: 

'''

summarize_prompt = '''
Резюмируй текст, кратко пересказывая и сохраняя важные детали. 
Пожалуйста, дай краткий обзор, выделив наиболее значимые моменты и обеспечив сохранение первоначального смысла и контекста: 
'''

found_info_1 = "Не удалось ничего найти"

save_emb_dialog_history = 'Краткая история диалога'

slash_n = '\n'

save_emb_dialog_mark_thesis_1 = 'Выдели тезисы из текста:\n'

save_emb_dialog_mark_thesis_2 = '\nКаждый тезис выделяй строкой !!!тезис!!!'

save_emb_dialog_thesis = '!!!тезис!!!'

save_emb_dialog_mark_group = 'Раздели диалог на смысловые группы. Между группами ставь строку !!!группа!!!.\n'

save_emb_dialog_group = '!!!группа!!!'

slash_token = '/'

dot_zpt_token = ';'

librarier_nothing_found = 'Не удалось ничего найти'

kv_token = '"'

prompt_n = ' Промпт:\n'

voskl_zn_token = '!!!'

tool_selector_return_1 = '\nФункция: '

tool_selector_return_2 = 'Неправильная команда'

start_dialog_tool_text = 'Прочитай промпт для оператора и напиши имена инструментов через запятую из имеющихся, которые нужны данному оператору для контроля результатов специалиста. Например: "tool_name1, tool_name2". Если ни одна не подходит, напиши: "None". Имеющиеся инструменты:\n'

# это можно объединить

system_role_text = 'Система: '

operator_role_text = '\nМилана: '

worker_role_text = '\nИван: '

func_role_text = '\nФункция: '

wrong_command = 'Неправильная команда'

start_dialog_history = 'Краткая история диалога: '

make_spec_first = 'Создай специалиста перед началом диалога'

udp_spec_if_needed = '\nОбнови специалиста, если нужно'

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

gigo_dreamer = 'мечтатель'
gigo_realist = 'реалист'
gigo_critic = 'критик'
gigo_questions = 'Напиши вопрос или вопросы, по одному на строку, какой информации нехватает для выполнения задачи: '
gigo_found_info = 'Доступна только следующая информация, другой не будет, запрашивать ещё или задавать вопросы тоже нельзя: '
gigo_not_found_info = 'Больше информации о задаче нет и не будет, запрашивать информацию или задавать вопросы тоже нельзя.'
gigo_role_answer_1 = 'Ты - '
gigo_role_answer_2 = '. Ответь, как сполна удовлетворить клиента, поставившего задачу: '
gigo_make_plan = 'Придумай план выполнения задачи на основе следующих данных:\nЗадача: '
gigo_return_1 = 'Задача'
gigo_return_2 = 'План'
gigo_reaction = 'Мысли разных людей об исполнении задачи:'

start_load_attachments_text = 'Происходит загрузка вложений, может занять много времени...'
end_load_attachments_text = 'Вложения загружены'

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
