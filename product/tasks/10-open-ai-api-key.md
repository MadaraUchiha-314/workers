- add a file called secrets.json
- add the file to .gitignore
- add a key for the open-ai-api-key
- add an optional argument to the constructor of an agent to take an API key
- read the secrets.json in the service layer `app.py` and pass it to the agent while initialization
- in the agent, if an API key is passed, use it, else use a random string like "ollama"
- use https://api.openai.com/v1/chat/completions as the default base url in the constructor
- pass the base url also from the service layer to the agent layer in the constructor and keep the ollama Open AI compatible as the default
- keep the default value of the API key also as "ollama" in the constructor

- add a settings.py in the service/ folder which contains all the settings for the app
- create different settings for each environment like local, e2e, prod etc derived from base settings
- move the LLM base url from app.py to settings
- move the api_key reading from secrets to settings

- like api key and base url, pass the model name from app.py to supervisor
- remove the default values of api_key, model and base url from the constructor of supervisor
