'''
skip
use this command if you were not trying to call any tool and just want to send your message to the user
'''
from cross_gpt import global_state, let_log

def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = ()
        let_log('ИНИЦИАЛИЗАЦИЯ')
        return
    global_state.stop_agent = True
    return text