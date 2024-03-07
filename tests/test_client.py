import json
import requests

res = requests.post('http://127.0.0.1:8080/v1/chat/completions',
                    json={
                        'messages': [{
                            "role": "user",
                            "content": "what is AI?"
                        }],
                        "stream": True,
                    },
                    stream=True)

if res.status_code == 200:
    for chunk in res.iter_content(chunk_size=1024):
        data = json.loads(chunk.decode()[6:])["choices"][0]
        if data["finish_reason"] == "stop":
            break
        print(data["delta"]["content"], end="", flush=True)
