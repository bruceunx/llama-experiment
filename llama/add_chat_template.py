import struct
import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUF_DEFAULT_ALIGNMENT

#
file_path = "../openhermes.gguf"
new_file_path = "../add_chat_model.gguf"
#
reader = GGUFReader(file_path, "r+")
#

# get chat template
CHAT_TEMPLATE = "tokenizer.chat_template"
chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# generate and adjust struct pack value from chat template with alignment
alignment = GGUF_DEFAULT_ALIGNMENT
new_align = reader.fields.get('general.alignment')
if new_align is not None:
    alignment = new_align.parts[-1][0]

add_data = bytearray()

name_data = CHAT_TEMPLATE.encode("utf-8")
add_data += struct.pack("Q", len(name_data))
add_data += name_data
add_data += struct.pack("I", GGUFValueType.STRING.value)

raw_len = len(add_data) + 8 + len(chat_template)
add_len = alignment - (raw_len % alignment)
if add_len != 0:
    chat_template += " " * add_len

raw_data = chat_template.encode("utf-8")
add_data += struct.pack("Q", len(raw_data))
add_data += raw_data

# insert raw bytes into file
# find insert index
kv = reader.fields
last_field = list(kv.values())[-1]
insert_offset = last_field.offset

# copy original data
new_data = reader.data.copy()
new_data = np.concatenate(
    (new_data[:insert_offset], add_data, new_data[insert_offset:]))

# add kv_count
kv_count_idx = reader.fields["GGUF.kv_count"].parts[0][0]
new_data[kv_count_idx] += 1

# save file
with open(new_file_path, "wb") as file:
    file.write(new_data.tobytes())
