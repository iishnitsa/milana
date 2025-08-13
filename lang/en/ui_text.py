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
    "restart_required": "Application restart is required to apply language settings.",

    # Language Loading
    "lang_load_error_title": "Language Load Error",
    "lang_load_error_message": "No language files found in 'lang' folder. The application will now close.",

    # Main Window
    "new_chat": "New Chat",
    "settings": "Settings",
    "attachments": "Attachments:",
    
    "chat_prefix": "Chat ",

    # Chat Context Menu
    "active_chat_delete_title": "Active Chat",
    "active_chat_delete_message": "Chat '{chat_name}' is active. How to delete?",
    "active_chat_delete_detail": "You can safely stop it or force delete.",
    "delete_chat_confirm_title": "Delete Chat",
    "delete_chat_confirm_message": "Are you sure you want to delete chat '{chat_name}'?",

    # App Closing
    "active_chats_on_close_title": "Active Chats",
    "active_chats_on_close_message": "There are {count} active chats. Choose action:",
    "active_chats_on_close_detail": "You can safely stop them or force close.",
    "stop_safely": "Stop Safely",
    "terminate": "Force Close",

    # Chat Controls
    "message_from_model_requires_answer": "Requires your response",

    # Attachment
    "attachment_open_error": "Attachment Open Error",

    # Initial Settings
    "initial_settings_title": "Initial Setup",
    "save_and_continue": "Save and Continue",
    "validation_error": "Validation Error",
    "token_limit_info": "Token limit (max: {max_tokens})",
    "validate_model": "Validate Model",
    "model_validated_success": "Model validated successfully. Max tokens: {tokens}",
    "model_not_validated": "Model not validated",
    "model_not_validated_continue": "The model is not validated. Continue?",

    # Model Types and Settings
    "model_type": "Model Type:",
    "local_model_path": "Model Path:",
    "gguf_files": "GGUF Files",
    "huggingface_model": "HuggingFace Model:",
    "hf_token_placeholder": "chat=chat_model_repo;emb=embeddings_model_repo;token=token",
    "openai_api_key": "chat=chat_model;emb=embeddings_model;token=api_key",
    "anthropic_api_key": "chat=chat_model;emb=embeddings_model;token=api_key",
    "mistral_api_key": "token=api_key",
    "cohere_api_key": "chat=chat_model;emb=embeddings_model;token=api_key",
    "ollama_model": "host=localhost;port=11434;chat=chat_model;emb=embeddings_model;token=optional",
    "ollama_model_hint": "",
    "lmstudio_model_hint": "Enter identifier, e.g. 'local-model/model-name'",
    "lmstudio_model": "host=localhost;port=8080;chat=chat_model;emb=embeddings_model;user=optional;password=optional",
    "custom_api_server_ip": "Server IP:",
    "custom_api_port": "Port:",
    "custom_api_key": "API Key (if required):",
    "token_limit": "Token Limit:",

    # Create/Edit Chat Window
    "create_chat_title": "Create New Chat",
    "chat_name": "Chat Name:",
    "chat_name_exists": "Chat with this name already exists.",
    "enter_chat_name": "Enter chat name",
    "tab_model": "Model",
    "tab_chat_settings": "Chat Settings",
    "tab_modules": "Modules",
    "create": "Create",
    "max_tasks": "Max. tasks:",
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
    "settings_saved": "Settings saved successfully.",
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
    "model_err_path_missing": "Model path is required",
    "model_err_invalid_file": "Invalid model file",
    "model_err_validation_generic": "Model validation error: {e}",
    "custom_api_success": "API connection successful",
    "custom_api_fail": "Failed to connect to API",
    "custom_api_connect_error": "Connection error: {e}",
    "model_cfg_validated": "Settings validated successfully",
    "model_type_no_validation": "Model type doesn't require validation",
    "model_err_hf_missing": "HuggingFace model is required",
    "model_err_openai_key_missing": "Enter OpenAI API key",
    "model_err_anthropic_key_missing": "Enter Anthropic API key",
    "model_err_mistral_key_missing": "Enter Mistral API key",
    "model_err_cohere_key_missing": "Enter Cohere API key",
    "model_err_ollama_missing": "Ollama model is required",
    "model_err_lmstudio_missing": "LM Studio model is required",
    "model_err_custom_api_missing": "Server IP and port are required",
}
