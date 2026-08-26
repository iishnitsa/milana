'''
skip
fallback only: force a plain message when the system mistakes it for a command; not for normal chat
'''
from cross_gpt import global_state, let_log

def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = ()
        let_log('ИНИЦИАЛИЗАЦИЯ')
        return
    global_state.stop_agent = True
    return text