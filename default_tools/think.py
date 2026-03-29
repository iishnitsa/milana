'''
think
send to make the message not sent to the interlocutor, nothing else will happen, but it will give you an opportunity to think
Think
The module simply does not send the message to the interlocutor and does not break the agent's cycle. Experimental
'''

import cross_gpt # чтобы система посчитала функцию системной

def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'answer',
        )
        main.answer = 'Accepted. Your thought was not sent to the interlocutor.'
        return
    return main.answer