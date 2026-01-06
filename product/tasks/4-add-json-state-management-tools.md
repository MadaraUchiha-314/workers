- Add a new attribute of the agent state called `data` which is going to be fully managed by the agent through 2 tools
- Add 2 tools which help in management of the attribute `data` in the state
- One tool which is a wrapper on jsonpath which allows querying of any attribute of the `data` attribute using jsonpath
- One tool which is a wrapper over jsonpatch which allows arbitrary modification of any json path of the `data` attribute

- Implementation:
    - LangGraph tools get the state passed in as the argument using InjectedState and are able to modify the state
        - https://docs.langchain.com/oss/python/langchain/tools#accessing-context
- Install dependencies for jsonpath-ng and jsonpatch