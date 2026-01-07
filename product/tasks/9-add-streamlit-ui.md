- Add a streamlit chat UI to chat with our service
- The service is compliant with A2A protocol with a JSONRPC endpoint hosted at: 
    - http://localhost:8000/supervisor/
- Understand the A2A protocol specification for JSONRPC to get what the request and response schemas are

- NOTE: In the response that's sent by the agent, there's an artifact of name "Agent State" which contains attributes like `data` and `messages` 
    - Add a section in the streamlit ui where the agent state is displayed (overwrite existing) after each turn

The agent card is available at: http://localhost:8000/supervisor/.well-known/agent-card.json


## Example Request/Responses:
- Request:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "contextId": "c2920705-f3eb-4698-b617-cbf4b62de6e5",
      "parts": [
        {
          "type": "text",
          "text": "connect to expert"
        }
      ],
      "messageId": "<<$guid>>"
    },
    "configuration": {
      "blocking": true
    }
  },
  "metadata": {}
}

```
- Response:
```json
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": {
    "artifacts": [
      {
        "artifactId": "agent-state-60203f56-c690-44c8-98ad-b825e7c76cd4",
        "name": "Agent State",
        "parts": [
          {
            "data": {
              "messages": [
                {
                  "content": "connect to expert",
                  "additional_kwargs": {},
                  "response_metadata": {},
                  "type": "human",
                  "name": null,
                  "id": "af1bba70-406f-4a37-bb46-4c6d2446d909"
                },
                {
                  "content": "",
                  "additional_kwargs": {
                    "refusal": null
                  },
                  "response_metadata": {
                    "token_usage": {
                      "completion_tokens": 260,
                      "prompt_tokens": 810,
                      "total_tokens": 1070,
                      "completion_tokens_details": null,
                      "prompt_tokens_details": null
                    },
                    "model_provider": "openai",
                    "model_name": "gpt-oss:20b",
                    "system_fingerprint": "fp_ollama",
                    "id": "chatcmpl-714",
                    "finish_reason": "tool_calls",
                    "logprobs": null
                  },
                  "type": "ai",
                  "name": null,
                  "id": "lc_run--019b9988-e93e-7c90-9716-829ac5d43c3f-0",
                  "tool_calls": [
                    {
                      "name": "jsonpatch_update",
                      "args": {
                        "patch": "[{\"op\":\"add\",\"path\":\"/human_requested_count\",\"value\":1}]"
                      },
                      "id": "call_z2jzuufh",
                      "type": "tool_call"
                    }
                  ],
                  "invalid_tool_calls": [],
                  "usage_metadata": {
                    "input_tokens": 810,
                    "output_tokens": 260,
                    "total_tokens": 1070,
                    "input_token_details": {},
                    "output_token_details": {}
                  }
                },
                {
                  "content": "Successfully applied patch. New data: {\"human_requested_count\": 1}",
                  "additional_kwargs": {},
                  "response_metadata": {},
                  "type": "tool",
                  "name": "jsonpatch_update",
                  "id": "2777552f-6ba7-41d3-820f-88629a56aca1",
                  "tool_call_id": "call_z2jzuufh",
                  "artifact": null,
                  "status": "success"
                },
                {
                  "content": "I’m sorry, but I can’t connect you to an expert directly. However, I’m here to help with any questions or issues you have. Let me know what you need assistance with!",
                  "additional_kwargs": {
                    "refusal": null
                  },
                  "response_metadata": {
                    "token_usage": {
                      "completion_tokens": 143,
                      "prompt_tokens": 883,
                      "total_tokens": 1026,
                      "completion_tokens_details": null,
                      "prompt_tokens_details": null
                    },
                    "model_provider": "openai",
                    "model_name": "gpt-oss:20b",
                    "system_fingerprint": "fp_ollama",
                    "id": "chatcmpl-851",
                    "finish_reason": "stop",
                    "logprobs": null
                  },
                  "type": "ai",
                  "name": null,
                  "id": "lc_run--019b9989-04c1-7e30-8dea-794f98c34546-0",
                  "tool_calls": [],
                  "invalid_tool_calls": [],
                  "usage_metadata": {
                    "input_tokens": 883,
                    "output_tokens": 143,
                    "total_tokens": 1026,
                    "input_token_details": {},
                    "output_token_details": {}
                  }
                }
              ],
              "data": {
                "human_requested_count": 1
              }
            },
            "kind": "data"
          }
        ]
      }
    ],
    "contextId": "c2920705-f3eb-4698-b617-cbf4b62de6e5",
    "history": [
      {
        "contextId": "c2920705-f3eb-4698-b617-cbf4b62de6e5",
        "kind": "message",
        "messageId": "61c636a8-666e-4526-80f2-a3ef75b1cdb4",
        "parts": [
          {
            "kind": "text",
            "text": "connect to expert"
          }
        ],
        "role": "user",
        "taskId": "f49470f3-3a9e-4255-878d-de3f87e8f05e"
      }
    ],
    "id": "f49470f3-3a9e-4255-878d-de3f87e8f05e",
    "kind": "task",
    "status": {
      "message": {
        "contextId": "c2920705-f3eb-4698-b617-cbf4b62de6e5",
        "kind": "message",
        "messageId": "5aa97913-4677-4a17-bb38-1f56603434a9",
        "parts": [
          {
            "kind": "text",
            "text": "I’m sorry, but I can’t connect you to an expert directly. However, I’m here to help with any questions or issues you have. Let me know what you need assistance with!"
          }
        ],
        "role": "agent",
        "taskId": "f49470f3-3a9e-4255-878d-de3f87e8f05e"
      },
      "state": "input-required",
      "timestamp": "2026-01-07T17:37:34.123957+00:00"
    }
  }
}
```


- Updates:
    - Move the streamlit related code workers/streamlit folder
    - The UI looks very bad in dark mode making it very hard to read the text
    - fix the contrast and make it look good in both light and dark mode
    - add a vscode launch script to launch the streamlit ui. 
        - evaluate the need for `run.py`. why is it required when `app.py` already exists. if not needed, delete it.
    - the first message to an agent will not contain a `task_id`
    - if you receive a message from the agent and the task status is `input-required` then continue using the same `task_id` and `context_id` to communicate further with the agent
    - if you receive a message from the agent and the task status is a terminal task status, then use a new `task_id` for further comms
    - indicate the `task_id` in the streamlit ui