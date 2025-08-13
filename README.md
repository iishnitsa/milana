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

The project is very raw, but I decided to release it to avoid getting stuck in endless refinement. Versions will be labeled with the publication date.  
</details>  

<details>  
<summary>Installation</summary>  

**Windows**  
Before installation, you need:  
- Git  
- "Desktop development with C++" workload (via Visual Studio Installer)  
- Python 3.12 (not 3.13, as compilation issues may occur).  

Run `windows.bat` in the `install` folder and wait for the shortcuts to be created.  

**Linux and macOS**  
Run `linux_macos.sh` in the `install` folder.  
</details>  

<details>  
<summary>How to use</summary>  
1. Launch Milana and configure the model. Instruct models are recommended (e.g., Mistral Instruct).  
2. Choose a model provider (Ollama or HuggingFaceHub).  
   - For HuggingFaceHub, enter a string like:  
     `chat=mistralai/Mistral-7B-Instruct-v0.2;emb=sentence-transformers/all-MiniLM-L6-v2;token=hf_yourtoken`  
   - For Ollama, download the models (e.g., `mistral:latest` and `all-minilm:latest`). Enter the string:  
     `chat=mistral:latest;emb=all-minilm:latest`  
3. Click "Validate model" and save the settings.  
4. Enable the required modules in the settings (e.g., web search or command line).  
5. Create a chat, enter a task, and send the message.  

**Note:**  
- For stable operation, use powerful models or GPT-OSS.  
- If you encounter bugs, send logs and descriptions to Discord: `iishnitsa_milana`.  
</details>  

<details>  
<summary>For developers</summary>  

### How to develop modules  
A module consists of:  
1. A main file (e.g., `linux_cmd.py`).  
2. An optional localization file (e.g., `linux_cmd_lang.py`).  

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

Examples of modules can be found in the `default_tools` folder.  
</details>  

<details>  
<summary>Third-party modules</summary>  
Links to community-developed modules will appear here.  
</details>  

**Questions and suggestions:** Discord `iishnitsa_milana`.