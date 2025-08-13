'''
create_file
to create a file, enter the filename with extension, a comma, and the file contents. the file will be visible only to the client, and after creation, neither of you will have access to it
Create a file
Allows agent tos make different files — for now, you'll find them in the "files" folder inside the chat directory, until interface export is added
'''

import os
import uuid
from datetime import datetime
from cross_gpt import chat_path

def main(text: str) -> str:
    if not hasattr(main, 'attr_names'):
        main.attr_names = ('confirmation_text', 'error_no_comma', 'file_exists')
        main.confirmation_text = 'File saved'
        main.error_no_comma = 'Error: enter filename and contents separated by comma'
        main.file_exists = 'File with this name already exists, saved as: {filename}'
        return

    # Split into filename and content
    if ',' not in text:
        return main.error_no_comma
    
    filename_part, content = text.split(',', 1)
    filename = filename_part.strip()
    content = content.strip()

    base_dir = os.path.join(chat_path, "files")
    os.makedirs(base_dir, exist_ok=True)

    # Get filename parts
    name, ext = os.path.splitext(filename)
    counter = 1
    original_filename = filename
    
    # Find unique filename
    while os.path.exists(os.path.join(base_dir, filename)):
        filename = f"{name}_{counter}{ext}"
        counter += 1

    path = os.path.join(base_dir, filename)

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if filename != original_filename:
            return main.file_exists.format(filename=filename)
        return main.confirmation_text
    except Exception as e:
        print(f"Error: {e}")
        return main.confirmation_text