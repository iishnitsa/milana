'''
create_report
Just write its content, the file will be created automatically. The report will be saved in a folder accessible only to the real client who assigned the main task. When ending a dialog, still provide the result even if you've already recorded the same information in a report. Use this command to document found information, interesting discoveries, reasoning logic, encountered difficulties, and intermediate results during task execution. You can create multiple reports.
Creating a report file
Recommended if you want to record detailed results — for now, you'll find them in the "reports" folder inside the chat directory, until interface export is added
'''

import os
import re
from datetime import datetime
from cross_gpt import chat_path, send_output_message

def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = ('confirmation_text', 'report_created_text', 'empty_report_text')
        main.confirmation_text = 'Report saved'
        main.report_created_text = 'Report created: '
        main.empty_report_text = (
            'Report was not created: no content was provided after the command '
            '(only whitespace/newlines). Write the report body after !!!create_report!!!'
        )
        return

    body = (text or '').strip()
    if not body:
        return getattr(main, 'empty_report_text', None) or (
            'Report was not created: nothing was passed after the command.'
        )

    # Generate filename prefix from beginning of text
    raw_prefix = body[:50]  # take first 50 characters
    # Keep only alphanumeric, spaces, underscores; replace spaces with underscore
    safe_prefix = re.sub(r'[^\w\s]', '', raw_prefix)  # remove punctuation etc.
    safe_prefix = re.sub(r'\s+', '_', safe_prefix)    # replace spaces with underscore
    safe_prefix = safe_prefix.strip('_')              # trim leading/trailing underscores
    if not safe_prefix:
        safe_prefix = "report"
    else:
        # limit length to 30 characters
        safe_prefix = safe_prefix[:30]

    filename = f"{safe_prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    path = os.path.join(chat_path, "reports", filename)

    try:
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(body)
        # Маркер «для пользователя» — sidecar .for_user рядом с файлом
        try:
            with open(path + '.for_user', 'w', encoding='utf-8') as mf:
                mf.write('for_user=1\n')
        except Exception:
            pass
        # Cached UI send — resume must not re-push the same report bubble
        send_output_message(text=main.report_created_text + filename, attachments=[path])
    except Exception as e:
        print(f"Error: {e}")
        return f"Report error: {e}"

    return main.confirmation_text