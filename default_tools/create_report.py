'''
create_report
Just write its content, the file will be created automatically. The report will be saved in a folder accessible only to the real client who assigned the main task. When ending a dialog, still provide the result even if you've already recorded the same information in a report. Use this command to document found information, interesting discoveries, reasoning logic, encountered difficulties, and intermediate results during task execution. You can create multiple reports.
Creating a report file
Recommended if you want to record detailed results — for now, you'll find them in the "reports" folder inside the chat directory, until interface export is added
'''

import os
from datetime import datetime
from cross_gpt import chat_path, send_ui_no_cache

def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = ('confirmation_text', 'report_created_text')
        main.confirmation_text = 'Report saved'
        main.report_created_text = 'Report created: '
        return
    filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    path = os.path.join(chat_path, "reports", filename)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        send_ui_no_cache(main.report_created_text + filename, attach=[path])
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return main.confirmation_text