from jinja2.sandbox import ImmutableSandboxedEnvironment
from llama_cpp import Llama

llm = Llama("../new_file.gguf", n_gpu_layers=32, n_ctx=1024, verbose=False)

chat_temp = llm.metadata["tokenizer.chat_template"].strip()

_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
temp_eng = _env.from_string(chat_temp)

_messages = [{
    "role": "system",
    "content": "You are a very helpful AI assistant."
}]

while True:

    input_message = input("\nYou: ")
    if input_message == "q":
        break

    _messages.append({"role": "user", "content": input_message})

    prompts = temp_eng.render(messages=_messages, add_generation_prompt=True)

    output = llm.create_completion(
        prompt=prompts,
        temperature=0.7,
        stream=True,
        max_tokens=1000,
    )
    for result in output:
        print(result["choices"][0]['text'], end="", flush=True)  # type: ignore
