'''
ask_user
clarify user's intent or use them as information source
Ask the user
Milana will wait until she receives your response.
'''

from cross_gpt import get_input_message, send_output_message
from multiprocessing import Process, Pipe

def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = ('answer_text',)
        main.answer_text = 'Question to user: '
        return
    send_output_message(text=main.answer_text, command='ask_user')
    return get_input_message(command='answer_user')
    # должна провести диалог с пользователем и выдать суммарный ответ, мб даже получить файлы и скачать в ембеддинги, но пока это будет закомментировано, пока просто input
    # не забудь что это опционально
    # может будет не аск юсер а получить доп инфу а эта штука посмотрит на ембеддинги и подумает, брать оттуда/из инета/спросить у пользователя
    # придтся адаптировать worker под эту функцию