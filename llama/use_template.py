from llama_cpp import Llama

llm = Llama("./new_file.gguf", n_gpu_layers=32, n_ctx=1024, verbose=False)

breakpoint()
