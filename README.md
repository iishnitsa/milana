<p align="center"><img src="data/icons/icon.png" alt="Milana Logo" width="120"></p>
<h1 align="center"><span style="color: #5200ff;">Milana</span></h1>
<p align="center"><strong>Autonomous · Independent · Free · For Mere Mortals</strong></p>
<p align="center"><img src="https://img.shields.io/badge/Discord-iishnitsa_milana-5865F2?logo=discord&logoColor=white" alt="Discord"></p>

[Features (+video demo)](#features-and-use-cases) | [How It Works](#how-it-works) | [History](#history-and-project-details) | [Installation](#installation-for-users-or-in-venv-for-windowsmacoslinux-or-exe-build) | [How to Use](#how-to-use) | [Module Development](#how-to-develop-modules) | [Provider Development](#how-to-develop-model-providers) | [Third-Party Modules and Providers](#third-party-modules)

Here is what I am **trying** to achieve.

## Autonomous
It develops an execution plan, delegates complex tasks, and verifies itself without human intervention. Hierarchical delegation allows returning fully developed results instead of abstract answers for complex tasks. Using deep research? A single request of yours will trigger a chain of deep research and reasoning.

## Independent
Requires only a connection to a model, which can also run locally. Agent memory is stored locally. Computations happen locally.

## Free
No request limits, file upload limits, paid subscription tiers, or regional restrictions.

## For Mere Mortals
Just run the .exe installer; no console required.

## Optimized for Weak Models
Initially, I was limited by an old laptop, and then I grew to like it. I have to write detailed prompts and create special user-friendly conditions. It spends an incredibly small number of tokens on medium-complexity tasks. More to come.

## Universal
Supports any sufficiently smart instruct model. The model doesn't necessarily need agentic or tool-calling capabilities, nor does it need to strictly follow specific standards.

## The Best
The prototype is far from an attractive state, but I will constantly improve it, putting my soul and vision into it.

## Features and Use Cases
Module support allows the system to do anything, even turning on a kettle. At the moment, I've written a few modules; the truly useful ones are web search, deep research, and report generation. I haven't finished the command-line work yet.
You can write your own module using the documentation below. It's not difficult.
In the future, I will add compatibility with popular standards such as MCP.

The request I used to debug the system was related to progress in Alzheimer's treatment, as this ambiguous topic requires meticulous study. Literally, it was:
`
perform a meta-study on the entire study of alzheimer's and drugs for this disease, draw conclusions about what alzheimer's is according to the most likely theory (this can be found out by comparing many works), about scandals, about misconceptions, etc., in order to get the most reliable information about what alzheimer's is and how to treat it
periodically make reports on the information found and the conclusions drawn
`
Unfortunately, `ministral-3` responses are quite unstable, cloud `qwen3.5:2b` was unavailable, and there was no time for GPU-based work; the demo was recorded on `qwen3.5:cloud`, which isn't very cheap. In the future, I will disable thinking for the model in some places and record a demo on a weaker `Qwen3.5`.
With these settings, I obtained the following result, consisting of several reports and a final answer, though a demo with link-only access is better:
[Milana 04 2026 demo YouTube](https://www.youtube.com/watch?v=USj5WB6UfME)
Currently, interaction with the PC is limited. The command-line module doesn't inspire confidence, and direct advanced file handling hasn't been implemented. But! Both command line and file handling are in the plans. I won't say exactly what I'm going to do, but it will be a very clever and fault-tolerant system, also adapted for weak models.

## How It Works
```mermaid
flowchart TD
    A[Client Task] --> B[GIGO: Plan Generation - Dreamer → Realist → Critic]
    B --> C[Create Milana Operator and Select Tools]
    C --> D[Create Ivan Executor with Toolset]
    D --> E[Dialogue between Milana and Ivan, Task Execution]

    E --> F{Executor Delegates?}
    F -->|Yes| B
    F -->|No| G[End Dialogue and Transfer Result]

    G --> H[Critic Evaluates Result]
    H --> I{Satisfactory?}
    I -->|Yes, 3| J[Success → Result to Client or Ivan]
    I -->|No, <2 attempts| K[Formulate New Task for Executor or Operator]
    K --> B
    I -->|Unsure, 2| J
    I -->|No, 2 attempts| J
    J -->|Return to higher dialogue| E
```
The system begins by receiving a client task, which enters the GIGO block where three roles—Dreamer, Realist, and Critic—trigger sequentially to form a structured action plan. Based on this plan, a Milana agent-operator is created to select the necessary tools, followed by the creation of an Ivan executor with its own toolset, and a dialogue begins between them to execute the task. If during the dialogue the executor realizes it cannot handle the task, it can delegate it to a new level, returning the process to the GIGO block to create a nested dialogue. When the dialogue is finished and a result is obtained, it is passed to the Critic, who evaluates its quality (up to two attempts are given by default). Depending on the evaluation: on full success, the result is returned to the client; if unsure, human verification is required; if the result is unsatisfactory but attempts remain, a refined task is formed and the process returns to executor creation; if both attempts fail, the task with the critic's comments is returned to the level above (to the superior agent or the original client).

<details>
<summary>History and Project Details</summary>
One day I was talking to ChatGPT, asking for help in developing a project. In response, I received an implementation plan. At first, I fed the tasks from the plan to ChatGPT one by one, and then I had the idea to make it talk to itself.

Later I realized that the tasks were too complex for it, and it would be good to create a plan for those tasks as well. The idea expanded into a hierarchy that should grow by one level at the AI's command.

I proceeded to implementation.

I chose LangChain as the foundation. At the time, I thought it would be well-suited for creating an agent that would issue commands to create hierarchy levels. However, several problems arose during development:
1. LangChain changes constantly and significantly.
2. It is designed for powerful AI models, as even weak models can make mistakes when writing commands.
3. The library does not allow for fine integration into user code.

Therefore, I wrote my own mechanism.

I realized that not everyone has access to powerful AI, high-performance PCs, or a nuclear power plant to run servers. Weak models can also be useful if the right approach is found. For example, I allowed models to make typos in commands and tried to simplify the prompts.

**upd1**
While I was preparing the release for December 2025, I realized that a hierarchical structure is poorly suited for programming. I learned about this from here: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ and here: https://arxiv.org/abs/2512.08296.

I see a significant problem with data exchange between hierarchy levels and between dialogues within the same hierarchy level. I tried to solve this by saving the entire dialogue in embeddings after its completion so that a librarian could later extract information from it. I also need to figure out automatic safe creation, reading, editing, and deletion of files during operation, as well as automatic aggregation of file access. I will experiment with combining structures.

**upd2 05 2026**
The 05 2026 release is mostly bug fixes, minor refinements, and a redesign. I also worked hard to fix the data exchange problem. Now agents can get information about other dialogues, both existing and deleted. For the next release, I will focus on file handling and plan to make the hierarchy structure a bit more advanced, while also leaving room for non-hierarchical structures.

I removed "10,000 monkeys" and the best-solution selection. It was highly underdeveloped, very costly, and slow, which doesn't align with the philosophy of a cheap system on fast small models. I will decide to add it in the distant future.

My own command protocol and response system turned out to be even better than standard ones. True, I had to tinker with writing instructions for the model on correct command usage. Now the system doesn't depend on whether the model natively supports function calling. It doesn't depend on which calling standard the model was trained on. Standardization is a common problem among agents; sometimes the model tries to call a tool but triggers a parsing error due to a mismatch. Sometimes the agent simply refuses to work with the model. I still left the native call option but haven't tested it. In the future, I will work on debugging native calls and automatic standard detection, but for now, I recommend not enabling this option.

The project is very raw, but I decided to release it to avoid getting bogged down in endless refinement. I'll be happy to hear your ideas, bug reports, and suggestions. I have many interesting ideas for the future! Versions will be marked with the publication date.
</details>

<details>
<summary>Installation for Users or in Venv for Windows/macOS/Linux or .exe Build</summary>
**Installation for Users**

Just run `MilanaSetup.exe`. For Linux, unpack the Linux version, grant execution permission to the `Milana` file, and run it.

<details>
<summary>Installation Windows/macOS/Linux Venv, `.exe` or `ELF` Build</summary>
<details>
<summary>**Windows**</summary>
Before installation, you need:
- `Git`
- `MSVC Build Tools`
- `Python 3.13.7` (not `3.14`, as some libraries haven't been rewritten for `3.14`; I might handle the adaptation of my code later) with `Tk` installed (otherwise the interface won't work)

Run `windows.bat` in the `install` folder and wait for `start_milana.bat` to be created.
Or run `buildexe.bat` in the `install` folder and wait for `Milana.exe` to be created.
To create an installer, download Inno Setup and compile the installer using the `InnoSetupInstallerBuild.iss` config in the `install` folder.
</details>
<details>
<summary>**Linux and macOS**</summary>
Before installation, you need:
- pyenv
- python tk packages, e.g., `sudo pacman -S tk` (if you forgot to install them, clear the pyenv cache `pyenv uninstall 3.13.7`, install the packages, and try again), otherwise the interface won't work

Run `linux_macos.sh` in the `install` folder.
To compile a binary `ELF`, use the `build_linux.sh` script.
</details>
</details>
</details>

<details>
<summary>How to Use</summary>
1. Run Milana and configure the model. Models like `qwen3.5` are recommended.
2. Select a model provider (Ollama, GPT4All, or OpenAI). For Ollama, download the models (e.g., `qwen3.5:2b` and `all-minilm:latest`). Click `Validate model` and save the settings.
4. Enable the necessary modules in the settings.
6. Create a chat, enter a task, and send the message.

Chat Settings
- Hierarchy level limit: default 0, unlimited.
- Maximum critic reactions: default 2, 0 to disable. This is the final result evaluation option triggered at the end of any dialogue between agents at any hierarchy level. Its request for refinement recreates the dialogue with a new, refined task.
- Use advanced dialogue memory: enabled by default. Dialogue memory will use retrieval of relevant old messages using a vector database when the entire dialogue doesn't fit in the context. Similar to RAG. The mode where advanced memory is disabled is not programmed very well; disabling it is not recommended.
- Clear generations: disabled by default. Intended to help very small (1b or old 7b) models when the model violates the command call protocol or when generation breakdown occurs (the model starts generating a repeating phrase of several words many times or until the context runs out). It doesn't work very well yet and is usually not needed.
- Record log: enabled by default. Records almost every step in the code; quickly reaches 100 MB in a few hours. Disable it if you won't be analyzing or sharing the incorrect behavior of this prototype with me.
- Record results: disabled by default. If the prototype refuses to create reports despite the connected module, or if you just want to see the program's operation, enable it. The result of each completed dialogue between agents will be recorded.
- Use Librarian: disabled by default. A module that may sometimes work incorrectly. It searches through all internal data (from user files, past dialogues, and internet searches), as well as the internet if available. Usually, disabling it is not required.
- Recreate agents with a new task: disabled by default. When you receive a response in the chat, the last agent dialogue is already finished. Your response to the message will start a new dialogue with the initial task, the result, and your response. If you enable this parameter, the dialogue will not end and be recreated; your response will go to the operator in response to the dialogue completion request.
- Skip nested images: disabled by default. If an image is nested in an archive, docx, or pdf (including pdfs found by the system on the internet), it doesn't process it. Helps filter out junk images. But it might skip important information if, for example, text in a pdf is stored as a scanned image.

**Note:**
- For stable operation, use a model more powerful than the one whose result you didn't like. The system is not perfect, but `qwen3.5` with 7 or 2 billion parameters is usually enough.
- Only `Ollama` and `GPT4All` providers are well-tested.
- If you encounter errors, send to Discord `iishnitsa_milana`: screenshots/videos, `log.txt`, `cache.db`, `chatsettings.db`, and other relevant files from the chat folder where the problem occurred, with a detailed description.
</details>

<details>
<summary>How to Develop Modules</summary>
A module consists of:
- A main file (e.g., `linux_cmd.py`).
- An optional localization file with the same name ending in `_lang` in the same folder (e.g., `linux_cmd_lang.py`).

**Module Structure:**
```python
'''
# Command for the model (e.g., execute_command)
# Brief description for the model
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
Localization file (optional but recommended; you can also leave `main.` strings in the main file empty and move 'en' localization to the localization file):
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
Rules:
- The `main` function accepts and returns only text.
- Localization simplifies the work for both the model and the user.
- For text sent to the AI or user, use the `attr_names` structure; this is required for localization to work.
- The localization file must be in the same folder as the module.
- The `_lang` suffix for the localization file is mandatory.
- It is strongly recommended not to use third-party libraries or libraries not in `requirements.txt`. If necessary, ensure the module acts as a layer between your service (which will include third-party libraries) and Milana.
- You can use Milana's system functions, such as querying the LLM, with some caution.
- Consider the specifics of the cache system.

<details>
<summary>How to correctly use system functions **and work with system state saves for recovery after restart!**</summary>
Progress saving is done through caching:
- Critically important operations (changes in SQLite and ChromaDB databases, to avoid repeated modification or access to already modified or deleted data).
- Expensive and time-consuming operations (generation, web search, computations).
- Inside operations that change the system state, if they don't directly relate to changing the system state (e.g., `ask_model` inside `start_dialog`).

Caching is forbidden for operations:
- Changing the system state (e.g., `start_dialog`, so that during replay the system is brought to the state it was in at the time of shutdown).
- Located outside functions that change the system state, at any nesting level.
- Located inside or outside functions that use caching, at any nesting level; currently, the caching system does not support nesting.

The system caches many different types of data, except classes and functions. Exceptions will be saved to the cache and re-thrown.
The `pickle` library is used for serialization.

The caching system consists of:
- `read_cache`: reads the cache from the database by `id` from the `cache_counter` variable, returns `[True, deserialized_value]` where `deserialized_value` is the required value, then increments the counter by 1; otherwise returns `[False]` and does not increment the counter if the record is not found, meaning the cache has ended.
- `write_cache`: writes the value and increments the counter.
- `cacher`: uses `read_cache` and `write_cache` automatically; used via the `@cacher` decorator.

In most cases, you will only need `cacher`.

Use manual control with `read_cache` and `write_cache` only in case of extreme necessity and with great caution! Any violation of the sequence will lead to an immediate exception or the inability to continue work after shutdown.

Correct sequence:
- read/read when the cache is not exhausted.
- read/write when the cache is exhausted.

Always read first, then, if `[False]` is received, write.
**Example: read/read/read/read/write/read/write**

If you don't use system functions that use the decorator, don't use caching. In this case, the system will automatically cache the result of your tool's execution and, upon restart, use the cached result without running the tool.

If you use system functions that have a decorator but you need caching to avoid repeated computations or to avoid getting different data upon program restart, cache some actions separately, as in the example below.

Allowed:
```python
from cross_gpt import cacher, ask_model

@cacher
def get_weather(): return '+10, windy'

ask_model(get_weather())
```
Not allowed:
```python
from cross_gpt import cacher, ask_model

@cacher
def get_weather(): return ask_model('+10, windy')

get_weather()
```

If you don't need caching of the module's result or data inside it in any case, and the module must run upon system restart, write, for example, `from cross_gpt import *` or `from cross_gpt import ask_model`; the system will take this into account.

List of available and safe-to-use functions that have a decorator:
- `ask_model`: query the LLM.
- `get_embs`: generate embeddings.
- `text_cutter`: summarize and reduce text size.
- `get_input_message`: get a message from the user sent **after** calling this function.
- `send_output_message`: send a message to the chat with the user.
- `send_log_to_ui`: send a message to the log window.

Logging functions `let_log('text')` and `send_log_to_ui` are not decorated and do not affect the operation of saves; use them for debugging.
Descriptions of how to use all listed functions will be in the section below.
Everything else is at your own risk.
</details>
<details>
<summary>How to use some useful system functions</summary>
When writing your own modules, you can use built-in core functions. **All of them already have built-in caching**—you don't need to manually write `read_cache` / `write_cache` logic.

#### 1. `ask_model`
`ask_model(prompt_text, ...)`—the main function for generating responses with automatic fallback via `text_cutter` on context overflow.

**Basic Parameters**
* `prompt_text` (str, required): Request text.
* `system_prompt` (str, optional): System instruction.
* `all_user` (bool): If `True`, forces passing the entire context as the user (ignoring roles).
* `temperature` (float): Default 0.6.
* `limit` (int): Token limit (if supported).

**Dialogue Formation**
Pass a string assembled with markers to `prompt_text`. There should be an odd number of messages, not counting the system prompt. At the end, add a model marker (needed by the parser to determine roles); this marker should not be the same as the marker marking the first non-system message. The markers themselves are removed by the parser. Maintain message alternation.
Three system marker variables are defined in the core for this:
* `system_role_text`: system prompt marker (optional).
* `operator_role_text`: interlocutor 1 marker (e.g., user).
* `worker_role_text`: interlocutor 2 marker (e.g., model).

#### 2. Vector Memory (ChromaDB)
`get_embs(text)` returns a vector (list of floats) for the passed text. Returns [] on error. Text is automatically truncated on context overflow.

`coll_exec(action, coll_name, ...)` is a universal cached wrapper for any operations with vector collections.

Actions: add, update, delete, query, get, count, modify, delete_collection.
Collections (`coll_name`):
- `user_collection`: user files.
- `milana_collection`: system memory (dialogues, work results, web search results).
- `rag_collection`: stores message vectors for composing dialogues.

Filtering (`filters` parameter): Supports comparison operators and logical links. Used in `query` and `get`.
Operators:
- `$eq`, `$ne`: equal, not equal.
- `$gt`, `$gte`, `$lt`, `$lte`: comparison of numbers/strings.
- `$in`, `$nin`: inclusion in list / non-inclusion.
- `$and`, `$or`: logical groupings.

**Examples of Queries with Filters**

Search only among successful dialogues:
```python
result = coll_exec(
    action="query",
    coll_name="milana_collection",
    query_embeddings=[get_embs("alzheimer's treatment")],
    filters={"result": True},   # $eq by default
)
```
Exclude web sources:
```python
result = coll_exec(
    action="query",
    coll_name="user_collection",
    query_embeddings=[get_embs("latest news")],
    filters={"source": {"$ne": "web"}},
    n_results=5
)
```
Complex filter with `$and`:
```python
result = coll_exec(
    action="get",
    coll_name="milana_collection",
    filters={
        "$and": [
            {"done": "correct"},
            {"hierarchy": {"$contains": "/2:"}}   # example with partial string match
        ]
    },
    fetch="documents")
```
Filter by a list of values `$in`:
```python
result = coll_exec(
    action="query",
    coll_name="user_collection",
    query_embeddings=[get_embs("reports")],
    filters={"source": {"$in": ["file", "web"]}},
    n_results=8)
```
Important: When using `$in` or `$nin` for the `vector_id` field (i.e., filtering by document identifiers), `coll_exec` automatically switches to an efficient collection traversal algorithm. In other cases, filters work normally.

Example search with multiple conditions and metadata retrieval:
```python
result = coll_exec(
    action="query",
    coll_name="milana_collection",
    query_embeddings=[get_embs("plan criticism")],
    filters={"done": "incorrect", "dialog_type": "executor"},
    fetch=["documents", "metadatas", "distances"],
    n_results=3)
```
`result` will be a dictionary with keys 'documents', 'metadatas', 'distances'.

#### 3. Built-in Search Modules
The `librarian` and `simple_web_search` modules, although they don't have a decorator, use cached core functions internally. **Their calls also cannot be wrapped in the `@cacher` decorator at any nesting level** within your scripts, otherwise caching will break.

* `librarian`
    Intelligent database search. Searches first in system memory (`milana_collection`), then in user data (`user_collection`). If the internet is enabled and nothing is found, it automatically triggers a web search if available (if you've connected `simple_web_search.py` or another module ending in `_web_search.py`). Returns formatted snippets with sources.
    *Important rule for prompts:* When forming requests to the librarian, **be sure to put a question mark at the end of lines with questions**, as the module relies on them when parsing multi-line requests; a separate search is performed for each multi-line request, so requests must be self-contained.
    ```python
    from cross_gpt import librarian
    result = librarian("What are the latest studies on RAG architecture?\nWho is their author?\nHow to correctly apply RAG?")
    ```

* `simple_web_search`
    Direct search via DuckDuckGo. Accepts 1 query, collects raw website texts, compresses them, and returns a summary with links. Works only if web search is allowed in settings.
    ```python
    from cross_gpt import web_search
    result = web_search("python 3.12 release notes")
    ```
    You can write your own search module ending in `_web_search.py`, and `librarian` will use it.

#### 4. Auxiliary Utilities

* **`text_cutter(text, cut_message=False)`**: iterative compression of long reads via LLM.
    * `False`: brief summary (summarization).
    * `True`: detailed paraphrasing (without losing minor facts from the user's message).
* **`let_log(text)`**: output debugging information to the console and record it in `log.txt`.
* **`send_output_message(text=None, attachments=None)`**: send a message to the user interface.
* **`get_input_message()`**: wait for a message from the user sent after executing this command.
* **`send_log_to_ui(message)`**: send service text to the UI log window.
</details>
</details>

<details>
<summary>How to Develop Model Providers</summary>

Providers are scripts that teach Milana to communicate with different APIs (Ollama, OpenAI, Anthropic, etc.).
The system (`ui.py` and `cross_gpt.py`) reads the provider code directly (parses the file's AST tree), so the provider structure is strictly regulated. If the rules are violated, the provider won't even appear in the interface settings.

### 1. File and Import Rules
* **File Name:** Must be in the `model_providers` folder and end in `_provider.py` (e.g., `gemini_provider.py`). Must not start with an underscore `_`.
* **Dependencies:** Only built-in Python libraries (e.g., `json`, `re`, `time`) and the `requests` library are allowed. **Forbidden** to use third-party SDKs (like official `openai` or `anthropic` libraries), as users of compiled `.exe` versions won't be able to install them. All requests are made via raw `requests`.
* **Logging:** To output logs to the interface, import the system function: `from cross_gpt import let_log`.

### 2. Mandatory Global Variables
The system reads these variables directly from the module's namespace. They must be declared at the file level:

```python
# Default context limits
token_limit = 4095
emb_token_limit = 4095

# API capability flags
do_chat_construct = True  # Whether to use the Chat API (user/assistant roles). Almost always True.
native_func_call = False  # Whether the API supports native function calling (Tool calling). Usually False.

# Mandatory tags dictionary. If the API doesn't use explicit tags, leave them empty.
tags = {
    "bos": "", "eos": "",
    "sys_start": "", "sys_end": "",
    "user_start": "", "user_end": "",
    "assist_start": "", "assist_end": "",
    "tool_def_start": "", "tool_def_end": "",
    "tool_call_start": "", "tool_call_end": "",
    "tool_result_start": "", "tool_result_end": "",
}
```
### 3. Mandatory `connect` Function (The most important part!)
The application interface scans this function to dynamically build the settings menu. A `params` dictionary must be declared inside the function. The keys of this dictionary will become input fields in the UI.

If a key name contains the word `file`, `path`, or `dir`, the UI will automatically create a file selection button for it.
```python
def connect(connection_string, timeout=30, _decrypted_token=None):
    global token_limit, emb_token_limit, do_chat_construct, native_func_call, tags
    
    # ATTENTION: The params dictionary MUST be declared exactly like this.
    # ui.py parses this section of code to create fields in the settings!
    params = {
        "url": "http://api.example.com",
        "model": "example-model",
        "emb_model": "example-embed",
        "token": "", # UI will hide input with asterisks if the key contains token or password
    }

    # Parsing connection_string, which the UI will send after clicking "Save"
    for part in connection_string.split(";"):
        part = part.strip()
        if not part or "=" not in part: continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        if key in params:
            params[key] = value.strip()

    # The token may come encrypted from the UI (if a password is set),
    # then it is unpacked into _decrypted_token
    api_key = _decrypted_token if _decrypted_token else params["token"]

    try:
        # Here you initialize the requests session and check API availability
        # session = requests.Session() ...
        # response = session.get(...)
        
        # If successful, MUST return a list of 3 elements:
        # [Status (bool), Chat model token limit (int), Tags (dict)]
        return [True, token_limit, tags]
        
    except Exception as e:
        # If error, return 4 elements:
        # [Status (False), 0, Tags, "Error text"]
        return [False, 0, tags, f"Connection error: {e}"]
```
### 4. Mandatory Generation Functions
The system expects three functions for working with models. In them, it is necessary to correctly handle context overflow: if there are too many tokens, you must raise an exception with the string 'ContextOverflowError' in the text.
```python
def disconnect():
    """Closes the requests session and clears data."""
    # global session; if session: session.close(); session = None
    return True

def ask_model(generation_params):
    """
    Old completions method.
    Accepts a dictionary (contains the 'prompt' key).
    MUST return a string (str).
    """
    prompt = generation_params.get("prompt", "")
    # Make request to API...
    # Return text
    return "Generated text"

def ask_model_chat(generation_params):
    """
    Chat completions method.
    Accepts a dictionary (contains the 'messages' key).
    MUST return the FULL response dictionary from the API (dict).
    Parsing (searching for 'choices' or 'message') will be done by the system itself in cross_gpt.py.
    """
    messages = generation_params.get("messages", [])
    # Make request to API...
    # ...
    # If context is overflowed:
    # raise RuntimeError("ContextOverflowError")
    
    # Return raw JSON response from requests
    return {"choices": [{"message": {"content": "Model response"}}]}

def create_embeddings(text):
    """
    Embeddings generation method.
    MUST return a list of floating-point numbers (List[float]).
    """
    text = text.strip()
    # Make request to API...
    # ...
    # If context is overflowed:
    # raise RuntimeError("ContextOverflowError")
    return [0.01, -0.02, 0.05, ...]
```
### 5. Error Handling (Retry Logic)
Since you are using raw `requests`, you must independently implement retry logic (Exponential Backoff) for 429 (Rate Limit), 500+ (Server Errors), and network timeouts. Study the `_request_with_backoff` function in `ollama_provider.py` as a reference example.

Important exceptions expected by the engine:
- When API balance is exhausted: `raise RuntimeError("balance end")`
- On context overflow: `raise RuntimeError("ContextOverflowError")`
</details>

<details>
<summary>Third-Party Modules</summary>
Links to modules developed by the community will appear here.
</details>

<details>
<summary>Third-Party Providers</summary>
Links to providers developed by the community will appear here.
</details>