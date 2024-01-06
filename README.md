# LLMTestBed

This tool was developed to study the behavior of LLMs such as GPT-2 under adversarial conditions in order to analyze their propensity for generating incorrect facts.
While previous work often includes triggers in the prompt which contain non-english characters / unicode tokens etc, this tool constructs semantically meaningful prompts that can be inserted in different areas of a regular prompt which will trigger incorrect behavior from an LLM.

This tool also has a built in retrieval mechanism where it can search for relevant information using Google's Search Engine API and integrate it into a prompt.

# Credits
Autoprompt : https://github.com/ucinlp/autoprompt
This tool was motivated by AutoPrompt.


