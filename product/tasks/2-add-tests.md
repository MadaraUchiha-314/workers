- Understand the codebase in `workers/`
- Write tests to validate the features of the framework and the service sepearately
- framework tests look like unit tests
- service tests have both unit and integration tests
- integration tests should test only the final end-to-end service calls


### Tech Dedails
- Write comprehensive and detailed tests using pytest
- Tests should be runnable as a standalone file as well so that it's easy to debug using an editor, i.e. each test should contain:

```py
if __name__ == "__main__":
    ...
```

### Some request responses

#### Request-0
```
curl --request POST \
  --url http://localhost:8000/supervisor/ \
  --header 'X-Forwarded-Port: 8000' \
  --header 'content-type: application/json' \
  --data '{
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
          "text": "hi. what'\''s up ?"
        }
      ],
      "messageId": "53a733ae-9248-4b05-a823-77bb9f208d8f"
    },
    "metadata": {}
  }
}'
```

#### Response-0
```json
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": {
    "contextId": "c2920705-f3eb-4698-b617-cbf4b62de6e5",
    "history": [
      {
        "contextId": "c2920705-f3eb-4698-b617-cbf4b62de6e5",
        "kind": "message",
        "messageId": "ac919812-a655-4532-a525-5fbf9f2643db",
        "parts": [
          {
            "kind": "text",
            "text": "hi. what's up ?"
          }
        ],
        "role": "user",
        "taskId": "981b13ba-0e14-4c92-84d3-d76338d25188"
      }
    ],
    "id": "981b13ba-0e14-4c92-84d3-d76338d25188",
    "kind": "task",
    "status": {
      "message": {
        "contextId": "c2920705-f3eb-4698-b617-cbf4b62de6e5",
        "kind": "message",
        "messageId": "f70585c5-13db-4f11-a6cd-d3e09901fd31",
        "parts": [
          {
            "kind": "text",
            "text": "Hello, this is the supervisor agent!"
          }
        ],
        "role": "agent",
        "taskId": "981b13ba-0e14-4c92-84d3-d76338d25188"
      },
      "state": "completed",
      "timestamp": "2026-01-03T13:45:28.889629+00:00"
    }
  }
}
```
