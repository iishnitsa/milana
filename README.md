# milana  
A prototype of a dynamic hierarchical agent system focused on weak models and requiring minimal user intervention. The goal is to create something at the user's request, with the ability to interact with the physical world using various tools.  

<details>
<summary>About the project</summary>
Once, I was talking to ChatGPT, asking for help in developing a certain project. In response, I received an implementation plan. At first, I fed the tasks from the plan to ChatGPT one by one, and then I got the idea to make it communicate with itself.

Later, I realized that the tasks were too complex for it, and it would be good to create a plan for those tasks as well. The idea was expanded with a hierarchy that should grow by one level at the AI's command.

I started the implementation.

I chose LangChain as the foundation. At the time, I thought it would be well-suited for creating an agent that would issue commands to create hierarchy levels. However, several problems emerged during development:
1. LangChain changes constantly and significantly.
2. It is designed for powerful AI models, as even weak models can make mistakes in writing commands.
3. The library does not allow fine-grained integration into custom code.

So, I wrote my own mechanism.

I realized that not everyone has access to powerful AI, high-performance PCs, or a nuclear power plant to run servers. Weak models can also be useful if you find the right approach. For example, I allowed models to make typos in commands and tried to simplify prompts.

<details>
<summary>upd1</summary>
While I was preparing the release for December 2025, I realized that a hierarchical structure is poorly suited for programming. I learned about this from here: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ and from here: https://arxiv.org/abs/2512.08296.

I see a significant problem with data exchange between the levels of hierarchy and between dialogues within the same level of hierarchy. I tried to solve this by storing the entire dialogue in embeddings after it is completed, allowing the librarian to retrieve information from it later. I also need to figure out automatic safe creation, reading, editing, and deletion of files during work, as well as automatic aggregation of file access. I will experiment with combining structures.
</details>

The project is very raw, but I decided to release it to avoid getting stuck in endless refinement. I look forward to your ideas, bug reports, and suggestions. In the future, I have many interesting ideas! Versions will be labeled with the publication date.
</details>

<details>
<summary>Windows/macOS/Linux Venv installation or .exe Building</summary>

**Windows**

Before installation, you need:
- `Git`
- `Desktop development with C++` workload (via `Visual Studio Installer`)
- `Python 3.12` or latest `3.13` (not `3.14`, because some libraries were not rewrote for `3.14`)

Run `windows.bat` in the `install` folder and wait for the start_milana.bat to be created.

Or run `buildexe.bat` in the `install` folder and wait for the Milana.exe to be created.

To create the installer, download Inno Setup and compile the installer using the `InnoSetupInstallerBild.iss` config in the `install` folder.

**Linux and macOS**

Run `linux_macos.sh` in the `install` folder.
</details>

<details>
<summary>How to use</summary>
1. Launch Milana and configure the model. Instruct models are recommended (e.g., codegemma, Mistral Instruct or Ministral).
2. Choose a model provider (Ollama, GPT4All or OpenAI). For Ollama, download the models (e.g., `mistral:latest` and `all-minilm:latest`). Important: if the model name doesn't contain a colon, add `:latest` at the end.
3. Click "Validate model" and save the settings.
4. Enable the required modules in the settings (e.g., web search or command line).
5. Create a chat, enter a task, and send the message.

**Note:**
- For stable operation, use powerful models or `GPT-OSS`.
- Due to restrictions in Russia, I currently only support Ollama, GPT4All and OpenAI.
- If you encounter bugs, please send to Discord `iishnitsa_milana`: screenshots/videos, `log.txt`, `cache.db`, `chatsettings.db` and other relevant files from the chat folder where the issue occurred, along with a detailed description. My PC isn't powerful enough to reproduce all scenarios.
</details>

<details>
<summary>For developers</summary>

### Important Notes
- Due to technical limitations and restrictions in Russia, I currently only support Ollama, GPT4All and OpenAI.
- My PC is not powerful enough to test all scenarios. If you want me to make your changes to the program/find a bug/fix a bug, please write to me in Discord "iishnitsa_milana" with your suggestions, attach screenshots/videos, `log.txt` , `cache.db`, `chatsettings.db` and detailed descriptions.

<details>
<summary>How to develop modules</summary>
A module consists of:
- A main file (e.g., linux_cmd.py).
- An optional localization file (e.g., linux_cmd_lang.py).

**Module structure:**
```python
'''
# Command for the model (e.g. execute_command)
# Short description for the model 
# Module name for the user  
# Description for the user  
'''

def main(text: str) -> str:  
    if not hasattr(main, 'attr_names'):  
        main.attr_names = (  
            'output_text',  
            'forbidden_text',  
            'path_error_text',  
            'timeout_text',  
            'exception_text'  
        )  
        main.output_text = 'Output'  
        main.forbidden_text = 'Forbidden command detected'  
        main.path_error_text = 'Access to paths outside the workspace is forbidden'  
        main.timeout_text = 'Command timed out'  
        main.exception_text = 'Error:'  
        return  

    # Module logic  
    return "Result"  
```

**Localization file (optional but recommended):**  
```python  
locales = {  
    'ru': {  
        'module_doc': [  
            'command_for_model',  
            'description_for_model',  
            'name_for_user',  
            'description_for_user'  
        ],  
        'main.output_text': 'Output',  
        'main.forbidden_text': 'Forbidden command',  
        # other strings  
    }
}
```

**Rules:**
- The `main` function only accepts and returns text.
- Localization simplifies work for the model and the user.
- For text sent to the AI or user, use the `attr_names` structure.
- The localization file must be in the same folder as the module.
- The `_lang` suffix for the localization file is mandatory.
- It is highly desirable not to use third-party libraries or libraries that are not in `requirements.txt`. If it is necessary to do so, please make sure that the module acts as a layer between your service, which will include third-party libraries, and Milana.

Examples of modules can be found in the `default_tools` folder.
</details>

<details>
<summary>How to develop model providers</summary>
## Complete Guide to Writing Providers for CrossGPT

## File Structure and Requirements

### 1. File Naming
**MANDATORY:** `[service_name]_provider.py`
```
✅ openai_provider.py
✅ ollama_provider.py  
✅ gpt4all_provider.py
✅ myapi_provider.py
❌ openai.py
❌ ollama.py
❌ provider.py
```

### 2. Dependencies
**ONLY ALLOWED:**
- `requests` (already in `requirements.txt`)
- something if it also in `requirements.txt`
- Python standard library (`re`, `json`, `os`, `time`, etc.)

**PROHIBITED:**
- Any third-party libraries (`aiohttp`, `httpx`, `pydantic`, etc.)
- Libraries requiring compilation

**REASON:** Users of .exe version cannot install additional libraries.

## Mandatory Provider Structure

### Minimal Provider Template:
```python
# example_provider.py
import requests
import json
import re
import time
from typing import Dict, Any, Optional, List

# === GLOBAL VARIABLES ===
session: Optional[requests.Session] = None
base_url: str = ""
default_chat_model: Optional[str] = None

# Ollama for embeddings (fallback)
ollama_session: Optional[requests.Session] = None
ollama_base_url: str = "http://localhost:11434"
ollama_emb_model: str = "all-minilm:latest"
use_ollama_for_embeddings: bool = False

# Token limits
token_limit = 4095
emb_token_limit = 4095

# Operation modes
do_chat_construct = True     # Use chat/completions API
native_func_call = False     # Native function calling support

# Formatting tags (MANDATORY FORMAT)
tags = {
    "bos": "", "eos": "",
    "sys_start": "", "sys_end": "",
    "user_start": "", "user_end": "",
    "assist_start": "", "assist_end": "",
    "tool_def_start": "", "tool_def_end": "",
    "tool_call_start": "", "tool_call_end": "",
    "tool_result_start": "", "tool_result_end": "",
}

# Thinking filter (ONLY if API doesn't have native support)
filter_think_enabled: bool = False
filter_start_tag: str = "</think>"
filter_end_tag: str = ""

# === LOGGING ===
from cross_gpt import let_log

# === MANDATORY FUNCTIONS ===
```

## connect Function (mandatory)

```python
def connect(connection_string: str, timeout: int = 30) -> List[Any]:
    """
    Connect to API provider
    
    Connection string format:
    "url=http://api.example.com; model=my-model; emb_model=emb-model; 
     chat_template=True; native_func_call=False; 
     ollama_url=http://localhost:11434; ollama_emb_model=all-minilm:latest; 
     ollama=False; token_limit=8192; filter_think=False"
    
    ALL parameters are mandatory for UI support
    """
    global session, base_url, default_chat_model, token_limit, tags
    global ollama_session, ollama_base_url, ollama_emb_model, use_ollama_for_embeddings
    global do_chat_construct, native_func_call, filter_think_enabled, filter_start_tag, filter_end_tag
    
    from cross_gpt import let_log
    
    # 1. DEFAULT PARAMETERS (ALL MANDATORY)
    params = {
        "url": "http://localhost:8080",          # Base API URL
        "model": "default",                      # Chat model
        "emb_model": "text-embedding-ada-002",   # Embeddings model (if available)
        
        # === IMPORTANT: These parameters MUST be present ===
        "chat_template": "True",                 # "True"/"False" - use chat/completions
        "native_func_call": "False",             # "True"/"False" - function calling support
        "ollama_url": "http://localhost:11434",  # Ollama URL for embeddings fallback
        "ollama_emb_model": "all-minilm:latest", # Ollama model for embeddings
        "ollama": "False",                       # "True"/"False" - use Ollama for embeddings
        "token_limit": "4095",                   # Fallback token limit
        
        # Thinking filter (optional, but parameters must exist)
        "filter_think": "False",                 # "True"/"False" - filter think part
        "filter_start": "</think>",              # Start tag for think part
        "filter_end": "",                        # End tag for think part
    }
    
    # 2. PARSE CONNECTION STRING
    let_log(f"Parsing connection string: {connection_string}")
    
    for part in connection_string.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        
        try:
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key in params:
                params[key] = value
                let_log(f"  Parameter '{key}' = '{value}'")
        except:
            continue
    
    # 3. SET VARIABLES
    base_url = params["url"].rstrip("/")
    ollama_base_url = params["ollama_url"].rstrip("/")
    ollama_emb_model = params["ollama_emb_model"]
    
    # Boolean parameters
    do_chat_construct = params["chat_template"].lower() == "true"
    native_func_call = params["native_func_call"].lower() == "true"
    use_ollama_for_embeddings = params["ollama"].lower() == "true"
    filter_think_enabled = params["filter_think"].lower() == "true"
    
    filter_start_tag = params["filter_start"]
    filter_end_tag = params["filter_end"]
    
    # Numeric parameters
    try:
        token_limit = int(params["token_limit"])
    except:
        token_limit = 4095  # Fallback
    
    let_log(f"Provider settings:")
    let_log(f"  base_url: {base_url}")
    let_log(f"  do_chat_construct: {do_chat_construct}")
    let_log(f"  native_func_call: {native_func_call}")
    let_log(f"  use_ollama_for_embeddings: {use_ollama_for_embeddings}")
    let_log(f"  token_limit: {token_limit}")
    
    # 4. CONNECT TO MAIN API
    try:
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "CrossGPT-Provider/1.0"
        })
        
        # Test connection (depends on API)
        # For example, for OpenAI-compatible APIs:
        test_url = f"{base_url}/models"
        let_log(f"Testing connection to {test_url}")
        
        response = session.get(test_url, timeout=timeout)
        response.raise_for_status()
        
        # Get list of available models
        models_data = response.json()
        
        # Response format depends on API:
        # OpenAI: {"data": [{"id": "model1"}, {"id": "model2"}]}
        # Ollama: {"models": [{"name": "model1"}, {"name": "model2"}]}
        
        available_models = []
        if "data" in models_data:  # OpenAI format
            available_models = [model["id"] for model in models_data.get("data", [])]
        elif "models" in models_data:  # Ollama format
            available_models = [model["name"] for model in models_data.get("models", [])]
        
        let_log(f"Available models: {available_models}")
        
        if not available_models:
            return [False, 0, tags, "No models available on server"]
        
        # Select model
        requested_model = params.get("model")
        if requested_model and requested_model in available_models:
            default_chat_model = requested_model
        else:
            default_chat_model = available_models[0]
        
        let_log(f"Selected model: {default_chat_model}")
        
        # 5. AUTO-DETECT TOKEN LIMIT
        # Try to get model information
        try:
            # Depends on API - example for OpenAI-compatible
            model_info_url = f"{base_url}/models/{default_chat_model}"
            info_response = session.get(model_info_url, timeout=timeout)
            
            if info_response.status_code == 200:
                model_info = info_response.json()
                let_log(f"Model info: {json.dumps(model_info, indent=2)[:500]}...")
                
                # Try to find context limit in response
                # Different APIs store this in different places
                found_limit = find_context_size(model_info, base_url, {})
                if found_limit and found_limit > 100:
                    token_limit = found_limit
                    let_log(f"Auto-detected token_limit: {token_limit}")
        except Exception as e:
            let_log(f"Could not auto-detect token limit: {e}")
            # Use value from parameters
        
        # 6. CONNECT TO OLLAMA (for embeddings)
        if use_ollama_for_embeddings:
            try:
                ollama_session = requests.Session()
                ollama_session.headers.update({"Content-Type": "application/json"})
                
                # Check Ollama availability
                ollama_test = f"{ollama_base_url}/api/tags"
                ollama_response = ollama_session.get(ollama_test, timeout=10)
                
                if ollama_response.status_code == 200:
                    ollama_models = ollama_response.json().get("models", [])
                    ollama_model_names = [m["name"] for m in ollama_models]
                    
                    # Check if embeddings model is available
                    if ollama_emb_model not in ollama_model_names:
                        let_log(f"Model {ollama_emb_model} not found in Ollama, using first available")
                        if ollama_model_names:
                            ollama_emb_model = ollama_model_names[0]
                    
                    let_log(f"Ollama for embeddings available, model: {ollama_emb_model}")
                else:
                    let_log(f"Ollama not available (status {ollama_response.status_code})")
                    ollama_session = None
                    
            except Exception as e:
                let_log(f"Ollama connection error: {e}")
                ollama_session = None
        
        # 7. RETURN SUCCESS RESULT
        return [True, token_limit, tags]
    
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Could not connect to {base_url}: {e}"
        let_log(error_msg)
        return [False, 0, tags, error_msg]
    
    except requests.exceptions.Timeout as e:
        error_msg = f"Connection timeout to {base_url}"
        let_log(error_msg)
        return [False, 0, tags, error_msg]
    
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        let_log(error_msg)
        return [False, 0, tags, error_msg]
```

## Function for Auto-detecting Context Limit

```python
def find_context_size(model_data: Dict[str, Any], base_url: str, headers: Dict[str, str]) -> int:
    """
    Auto-detect context limit from model data.
    Searches in different possible places in API response.
    """
    from cross_gpt import let_log
    
    # 1. Try to get limit from server (if API supports it)
    try:
        config_url = f"{base_url}/api/config"
        config_resp = requests.get(config_url, headers=headers, timeout=10)
        if config_resp.status_code == 200:
            server_config = config_resp.json()
            server_limit = server_config.get('max_context_length')
            if server_limit is not None:
                let_log(f"Found limit in server config: {server_limit}")
                return int(server_limit)
    except:
        pass
    
    # 2. Search in possible JSON paths
    possible_paths = [
        ['parameters', 'num_ctx'],
        ['parameters', 'context_length'],
        ['model_info', 'context_length'],
        ['model_info', 'max_seq_len'],
        ['model_info', 'n_ctx'],
        ['details', 'context_length'],
        ['config', 'max_position_embeddings'],
        ['max_tokens'],
        ['context_length'],
    ]
    
    for path in possible_paths:
        try:
            value = model_data
            for key in path:
                value = value[key]
            
            if isinstance(value, (int, float)):
                let_log(f"Found limit in path {path}: {value}")
                return int(value)
        except (KeyError, TypeError):
            continue
    
    # 3. Search in string parameters
    if isinstance(model_data.get('parameters'), str):
        param_str = model_data['parameters']
        for line in param_str.split('\n'):
            if any(kw in line.lower() for kw in ['num_ctx', 'context', 'n_ctx', 'max_tokens']):
                numbers = re.findall(r'\b\d{3,5}\b', line)
                if numbers:
                    let_log(f"Found limit in string parameters: {numbers[-1]}")
                    return int(numbers[-1])
    
    # 4. Search numbers in JSON
    context_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 128000, 200000]
    model_text = json.dumps(model_data)
    found_sizes = sorted([int(num) for num in re.findall(r'\b\d{4,6}\b', model_text)
                         if int(num) in context_sizes], reverse=True)
    
    if found_sizes:
        let_log(f"Found limit in JSON text: {found_sizes[0]}")
        return found_sizes[0]
    
    # 5. Fallback
    let_log("Could not determine context limit, using 4095")
    return 4095
```

## ask_model Function (completions API)

```python
def ask_model(generation_params):
    """
    Text generation via completions API (/v1/completions)
    Returns string with response text
    """
    from cross_gpt import let_log
    
    # 1. CHECK CONNECTION
    if not session or not base_url:
        let_log("ERROR: Provider not connected")
        raise RuntimeError("Provider not connected. Call connect() first.")
    
    # 2. PREPARE URL
    api_url = f"{base_url}/v1/completions"
    let_log(f"ask_model: Sending request to {api_url}")
    
    # 3. ADD DEFAULT MODEL
    if 'model' not in generation_params and default_chat_model:
        generation_params['model'] = default_chat_model
        let_log(f"ask_model: Added default model: {default_chat_model}")
    
    # 4. SEND REQUEST
    try:
        let_log(f"ask_model: Request parameters: {json.dumps(generation_params, indent=2)[:500]}...")
        
        response = session.post(api_url, json=generation_params, timeout=60)
        
        # 5. PROCESS RESPONSE
        if response.status_code != 200:
            error_text = response.text
            let_log(f"ask_model: HTTP error {response.status_code}: {error_text}")
            
            # CRITICALLY IMPORTANT: ContextOverflowError handling
            if response.status_code in [413]:  # 413 - Payload Too Large
                let_log("ask_model: Error 413 -> ContextOverflowError")
                raise RuntimeError('ContextOverflowError')
            
            elif response.status_code == 400:  # 400 - Bad Request
                if any(keyword in error_text.lower() for keyword in ['context', 'length', 'token', 'exceed']):
                    let_log("ask_model: Error 400 is context-related -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                else:
                    raise RuntimeError(f"Request error: {error_text}")
            
            elif response.status_code == 500:  # 500 - Internal Server Error
                if 'context' in error_text.lower():
                    let_log("ask_model: Error 500 is context-related -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                else:
                    raise RuntimeError(f"Server error: {error_text}")
            
            elif response.status_code == 429:  # 429 - Too Many Requests
                raise RuntimeError(f"Too many requests (429): {error_text}")
            
            else:
                raise RuntimeError(f"HTTP error {response.status_code}: {error_text}")
        
        # 6. PARSE SUCCESSFUL RESPONSE
        data = response.json()
        let_log(f"ask_model: Received response, size: {len(str(data))} characters")
        
        if "choices" not in data or not data["choices"]:
            let_log("ask_model: Invalid response format - no choices")
            raise RuntimeError("Invalid response format - no choices")
        
        choice = data["choices"][0]
        result = choice.get("text", "").strip()
        
        # 7. THINK-PART FILTERING (if enabled)
        if filter_think_enabled:
            result = apply_think_filter(result)
        
        let_log(f"ask_model: Result: '{result[:100]}...'")
        return result
        
    except requests.exceptions.ConnectionError as e:
        let_log(f"ask_model: Connection error: {e}")
        raise RuntimeError(f"Connection error: {e}")
        
    except RuntimeError as e:
        # Re-raise ContextOverflowError and other RuntimeErrors as-is
        raise e
        
    except Exception as e:
        let_log(f"ask_model: Unexpected error: {e}")
        raise RuntimeError(f"Unexpected error: {e}")
```

## ask_model_chat Function (chat/completions API)

```python
def ask_model_chat(generation_params):
    """
    Text generation via chat/completions API (/v1/chat/completions)
    Returns FULL API response as dictionary (doesn't extract text)
    """
    from cross_gpt import let_log
    
    # 1. CHECK CONNECTION
    if not session or not base_url:
        let_log("ERROR: Provider not connected")
        raise RuntimeError("Provider not connected. Call connect() first.")
    
    # 2. PREPARE URL
    api_url = f"{base_url}/v1/chat/completions"
    let_log(f"ask_model_chat: Sending request to {api_url}")
    
    # 3. ADD DEFAULT MODEL
    if 'model' not in generation_params and default_chat_model:
        generation_params['model'] = default_chat_model
        let_log(f"ask_model_chat: Added default model: {default_chat_model}")
    
    # 4. SEND REQUEST
    try:
        let_log(f"ask_model_chat: Request parameters: {json.dumps(generation_params, indent=2)[:500]}...")
        
        response = session.post(api_url, json=generation_params, timeout=60)
        
        # 5. PROCESS RESPONSE
        if response.status_code != 200:
            error_text = response.text
            let_log(f"ask_model_chat: HTTP error {response.status_code}: {error_text}")
            
            # CRITICALLY IMPORTANT: ContextOverflowError handling
            if response.status_code in [413]:  # 413 - Payload Too Large
                let_log("ask_model_chat: Error 413 -> ContextOverflowError")
                raise RuntimeError('ContextOverflowError')
            
            elif response.status_code == 400:  # 400 - Bad Request
                if any(keyword in error_text.lower() for keyword in ['context', 'length', 'token', 'exceed']):
                    let_log("ask_model_chat: Error 400 is context-related -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                else:
                    raise RuntimeError(f"Request error: {error_text}")
            
            elif response.status_code == 500:  # 500 - Internal Server Error
                if 'context' in error_text.lower():
                    let_log("ask_model_chat: Error 500 is context-related -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                else:
                    raise RuntimeError(f"Server error: {error_text}")
            
            elif response.status_code == 429:  # 429 - Too Many Requests
                raise RuntimeError(f"Too many requests (429): {error_text}")
            
            else:
                raise RuntimeError(f"HTTP error {response.status_code}: {error_text}")
        
        # 6. PARSE SUCCESSFUL RESPONSE
        data = response.json()
        let_log(f"ask_model_chat: Received response, size: {len(str(data))} characters")
        
        # IMPORTANT: Return FULL response, system will extract text
        return data
        
    except requests.exceptions.ConnectionError as e:
        let_log(f"ask_model_chat: Connection error: {e}")
        raise RuntimeError(f"Connection error: {e}")
        
    except RuntimeError as e:
        # Re-raise ContextOverflowError and other RuntimeErrors as-is
        raise e
        
    except Exception as e:
        let_log(f"ask_model_chat: Unexpected error: {e}")
        raise RuntimeError(f"Unexpected error: {e}")
```

## create_embeddings Function

```python
def create_embeddings(text: str) -> List[float]:
    """
    Create vector embeddings for text.
    Priority: Main API -> Ollama -> Error
    """
    from cross_gpt import let_log
    
    let_log(f"create_embeddings: Received text, length {len(text)} characters")
    
    # 1. TRY MAIN API (if supported)
    if session and base_url:
        try:
            let_log(f"create_embeddings: Trying main API")
            
            # Format depends on API
            # OpenAI: /embeddings
            # Others: may have different endpoints
            
            # Check different possible endpoints
            endpoints = [
                f"{base_url}/embeddings",
                f"{base_url}/v1/embeddings",
                f"{base_url}/api/embeddings"
            ]
            
            for api_url in endpoints:
                try:
                    payload = {
                        "input": text,
                        "model": "text-embedding-ada-002"  # or from parameters
                    }
                    
                    let_log(f"create_embeddings: Trying {api_url}")
                    response = session.post(api_url, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Different response formats
                        if "data" in data and len(data["data"]) > 0:
                            embedding = data["data"][0]["embedding"]
                        elif "embedding" in data:
                            embedding = data["embedding"]
                        elif "embeddings" in data and len(data["embeddings"]) > 0:
                            embedding = data["embeddings"][0]
                        else:
                            continue  # Try next endpoint
                        
                        let_log(f"create_embeddings: Success from main API, size: {len(embedding)}")
                        return embedding
                        
                except Exception as e:
                    let_log(f"create_embeddings: Error with endpoint {api_url}: {e}")
                    continue
            
        except Exception as e:
            let_log(f"create_embeddings: Main API doesn't support embeddings: {e}")
    
    # 2. FALLBACK TO OLLAMA (if enabled)
    if use_ollama_for_embeddings and ollama_session and ollama_base_url:
        try:
            let_log(f"create_embeddings: Trying Ollama")
            
            api_url = f"{ollama_base_url}/api/embeddings"
            payload = {
                "model": ollama_emb_model,
                "prompt": text
            }
            
            response = ollama_session.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "embedding" in data:
                embedding = data["embedding"]
                let_log(f"create_embeddings: Success from Ollama, size: {len(embedding)}")
                return embedding
            else:
                let_log(f"create_embeddings: Ollama returned invalid format: {data}")
                
        except Exception as e:
            let_log(f"create_embeddings: Ollama error: {e}")
    
    # 3. IF NOTHING WORKED
    error_msg = "Failed to get embeddings. Check:\n1. API connection\n2. Embeddings model availability\n3. Ollama availability (if used)"
    let_log(f"create_embeddings: {error_msg}")
    raise RuntimeError(error_msg)
```

## Helper Functions

### Thinking Filter (only if API doesn't support natively)
```python
def apply_think_filter(text: str) -> str:
    """
    Filter think part from response.
    Used ONLY if API doesn't have native thinking support.
    """
    if not filter_think_enabled:
        return text
    
    from cross_gpt import let_log
    
    let_log(f"apply_think_filter: Applying filter. Start='{filter_start_tag}', End='{filter_end_tag}'")
    
    # Find start tag
    start_pos = text.find(filter_start_tag)
    
    if start_pos == -1:
        let_log("apply_think_filter: Start tag not found, returning original")
        return text
    
    # Take text after start tag
    filtered_text = text[start_pos + len(filter_start_tag):]
    let_log(f"apply_think_filter: Found start tag at position {start_pos}")
    
    # If end tag specified
    if filter_end_tag and filter_end_tag.strip():
        end_pos = filtered_text.find(filter_end_tag)
        if end_pos != -1:
            filtered_text = filtered_text[:end_pos]
            let_log(f"apply_think_filter: Cut to end tag at position {end_pos}")
        else:
            let_log("apply_think_filter: End tag not found, leaving as is")
    
    result = filtered_text.strip()
    let_log(f"apply_think_filter: Final text: '{result[:100]}...'")
    return result
```

### disconnect Function
```python
def disconnect() -> bool:
    """
    Close all connections and clean up resources.
    """
    global session, ollama_session, base_url, default_chat_model
    
    from cross_gpt import let_log
    
    let_log("disconnect: Closing connections")
    
    if session:
        session.close()
        session = None
        let_log("disconnect: Main session closed")
    
    if ollama_session:
        ollama_session.close()
        ollama_session = None
        let_log("disconnect: Ollama session closed")
    
    base_url = ""
    default_chat_model = None
    
    let_log("disconnect: All connections closed")
    return True
```

## Recursive Recovery on Connection Loss

```python
# Additional recovery logic (optional but recommended)

def ask_model_with_retry(generation_params):
    """
    Wrapper with recursive recovery on ConnectionError
    """
    if not hasattr(ask_model_with_retry, 'retry_count'):
        ask_model_with_retry.retry_count = 0
    
    try:
        result = ask_model(generation_params)
        ask_model_with_retry.retry_count = 0  # Reset on success
        return result
        
    except requests.exceptions.ConnectionError as e:
        ask_model_with_retry.retry_count += 1
        
        if ask_model_with_retry.retry_count > 3:  # Max 3 attempts
            ask_model_with_retry.retry_count = 0
            raise RuntimeError(f"Failed to restore connection after 3 attempts: {e}")
        
        let_log(f"Connection error, attempt {ask_model_with_retry.retry_count}/3 in 60 seconds")
        time.sleep(60)
        
        # Recursive retry
        return ask_model_with_retry(generation_params)
```

## Checklist Before Submission

### Mandatory Checks:
- [ ] File named `[name]_provider.py`
- [ ] No third-party libraries (only requests + standard)
- [ ] All 5 mandatory functions present
- [ ] `tags` has correct format
- [ ] All mandatory parameters in `connect()` supported
- [ ] `ContextOverflowError` handling exists
- [ ] `let_log` used for logging
- [ ] Ollama fallback for embeddings exists (highly recommended)
- [ ] Thinking filter added ONLY if API doesn't support natively
- [ ] Functions return correct types:
  - `connect()` → `List[Any]`
  - `ask_model()` → `str`
  - `ask_model_chat()` → `Dict` (full API response)
  - `create_embeddings()` → `List[float]`
  - `disconnect()` → `bool`

### Testing:
- [ ] Connection works with test string
- [ ] Text generation works
- [ ] Chat completions returns full response
- [ ] Embeddings created (main API or Ollama)
- [ ] Disconnect properly closes connections
- [ ] Network error handling works
- [ ] ContextOverflowError correctly detected

## Examples for Different APIs

### For OpenAI-compatible APIs:
```python
# API URL: https://api.openai.com/v1
# Endpoints: /completions, /chat/completions, /embeddings
# Models: gpt-4o, gpt-4-turbo, text-embedding-3-small
```

### For Ollama:
```python
# API URL: http://localhost:11434
# Endpoints: /api/generate, /api/chat, /api/embeddings
# Models: mistral:latest, llama3.2:latest, all-minilm:latest
```

### For Local Servers (LM Studio, Text Generation WebUI):
```python
# API URL: http://localhost:1234
# Endpoints: /v1/completions, /v1/chat/completions
# May not support embeddings → need Ollama fallback
```

## Common Issues

### 1. API Doesn't Support Embeddings
```python
# In create_embeddings() immediately switch to Ollama:
def create_embeddings(text: str) -> List[float]:
    if use_ollama_for_embeddings:
        # use Ollama
    else:
        raise RuntimeError("This API doesn't support embeddings. Enable Ollama fallback.")
```

### 2. API Has Non-Standard Endpoints
```python
# In connect() check available endpoints:
endpoints_to_check = ["/completions", "/v1/completions", "/api/generate"]
for endpoint in endpoints_to_check:
    try:
        response = session.get(f"{base_url}{endpoint}")
        if response.status_code == 200:
            # found working endpoint
            break
    except:
        continue
```

### 3. API Requires Authentication
```python
# In connect() add headers:
api_key = params.get("api_key") or os.getenv("API_KEY")
if api_key:
    session.headers.update({"Authorization": f"Bearer {api_key}"})
```

</details>
</details>

<details>
<summary>Third-party modules</summary>
Links to community-developed modules will appear here.
</details>

**Questions and suggestions:** Discord `iishnitsa_milana`