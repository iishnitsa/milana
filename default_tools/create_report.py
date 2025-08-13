'''
create_report
to create a report file, just write its contents, the file will be visible only to the client, and after creation, neither of you will have access to it
Creating a report file
Recommended if you want to record detailed results — for now, you'll find them in the "reports" folder inside the chat directory, until interface export is added
'''

import os
import uuid
from datetime import datetime
from cross_gpt import chat_path

def main(text: str) -> str:
    if not hasattr(main, 'attr_names'):
        main.attr_names = ('confirmation_text',)
        main.confirmation_text = 'Report saved'
        return

    base_dir = os.path.join(chat_path, "reports")
    os.makedirs(base_dir, exist_ok=True)

    # имя файла: timestamp + UUID.txt
    filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{uuid.uuid4().hex[:8]}.txt"
    path = os.path.join(base_dir, filename)

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        return main.confirmation_text
    except Exception as e:
        print(f"Error: {e}"); return main.confirmation_text