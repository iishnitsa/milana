# lang/en/ui_text.py
"""
Dictionary with text variables for English language.
"""
TEXTS = {
    # App General
    "app_title": "Milana",
    "error": "Error",
    "success": "Success",
    "warning": "Warning",
    "info": "Information",
    "save": "Save",
    "cancel": "Cancel",
    "close": "Close",
    "browse": "Browse...",
    "delete": "Delete",
    "language": "Language",

    # Language Loading
    "lang_load_error_title": "Language Load Error",
    "lang_load_error_message": "No language files found in 'lang' folder. The application will now close.",

    # Main Window
    "new_chat": "New Chat",
    "settings": "Settings",

    # Chat Context Menu
    "active_chat_delete_title": "Active Chat",
    "active_chat_delete_message": "Chat '{chat_name}' is active. Delete?",
    "delete_chat_confirm_title": "Delete Chat",
    "delete_chat_confirm_message": "Are you sure you want to delete chat '{chat_name}'?",

    # App Closing
    "active_chats_on_close_title": "Active Chats",
    "active_chats_on_close_message": "There are {count} active chats. Quit?",

    # Attachment
    "attachments": "Attachments",
    "attachment_open_error": "Attachment Open Error",

    # Initial Settings
    "initial_settings_title": "Initial Setup",
    "save_and_continue": "Save and Continue",
    "validation_error": "Validation Error",
    "token_limit_info": "Token limit",
    "validate_model": "Validate Model",
    "model_validated_success": "Model validated successfully. Max tokens: {tokens}",
    "model_not_validated": "Model not validated",
    "model_not_validated_continue": "The model is not validated. Continue?",

    # Create/Edit Chat Window
    "create_chat_title": "Create New Chat",
    "chat_name": "Chat Name:",
    "chat_name_exists": "Chat with this name already exists.",
    "enter_chat_name": "Enter chat name",
    "tab_model": "Model",
    "tab_chat_settings": "Chat Settings",
    "tab_modules": "Modules",
    "create": "Create",
    "max_critic_reactions": "Max. reactions of the critic:",
    "use_rag": "Use advanced dialogue memory",
    "filter_generations": "Purify the generation",
    "hierarchy_limit": "Hierarchy level limit",
    "use_librarian": "Use Librarian",
    "recreate_agents": "Recreate agents with a new task",
    "skip_nested_images": "Skip nested images",
    "system_modules": "System Modules",
    "global_custom_modules": "Custom Modules (Global)",
    "chat_specific_modules": "New Modules (Chat-specific)",
    "add_module": "Add Module",
    "add_new_module_button": "+",
    "module_validation_error": "Module validation failed:\n{error_msg}",
    "python_files": "Python Files",

    # Global Settings Window
    "settings_title": "Settings",
    "tab_main": "Main",
    "reset_settings_button": "Reset Settings",
    "reset_settings_confirm_title": "Reset Settings",
    "reset_settings_confirm_message": "Are you sure you want to reset all settings? The application will close.",
    "select_chat_to_configure": "Select a chat to configure.",
    "remove_module_confirm": "Remove selected module?",
    "error_adding_module": "Error adding module",

    # Module Validator
    "module_err_not_found": "File not found: {path}",
    "module_err_no_docstring": "Module must contain a docstring",
    "module_err_docstring_len": "Docstring must be at least 4 lines long",
    "module_err_main_not_found": "Module must contain a main function",
    "module_err_main_args": "Main function must take exactly 1 argument",
    "module_err_syntax": "Syntax error: {e}",
    "module_err_generic": "Module validation error: {e}",
    "module_validated": "Module validated successfully",
    "module_custom_desc": "Custom Module",
    "module_desc_missing": "Description missing",

    # Model Validator
    "model_err_validation_generic": "Model validation error: {e}",
    
    # Model Validator (added)
    "model_err_no_provider": "No model provider selected",
    "model_err_provider_missing": "Provider '{provider}' not found",

    "write_log": "Record a log",
    "write_results": "Record results",

    # Text Editor Context Menu
    "cut": "Cut",
    "copy": "Copy",
    "paste": "Paste",
    "select_all": "Select All",

    # File Dialog
    "all_files": "All files",

    # Folder Operations
    "folder_not_found": "Folder '{folder_name}' not found",
    "open_folder_error": "Error opening folder: {e}",

    # Providers
    "model_type": "Providers",
    "token_limit": "Token limit",
    "no_providers_found": "No providers found",

    # Settings
    "restart_required": "Restart required for changes to take effect",

    "ok": "OK",
    "yes": "Yes",
    "no": "No",
}