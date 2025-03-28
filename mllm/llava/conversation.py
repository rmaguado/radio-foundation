import dataclasses
from enum import auto, Enum
from typing import List

from mllm.llava.constants import DEFAULT_IMAGE_TOKEN


class SeparatorStyle(Enum):
    """Different separator style."""

    PLAIN = auto()
    LLAMA_3 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle
    multimodal: bool = False

    def parse_plain(self, messages):
        prompt = self.system
        if self.multimodal:
            prompt += f"<Image>{DEFAULT_IMAGE_TOKEN}</Image>"
        for i, (role, message) in enumerate(messages):
            if message:
                prompt += "###" + message + "\n"
        return prompt

    def parse_llama_3(self, messagse):
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.system}<|eot_id|>"
        if self.multimodal:
            prompt += f"<Image>{DEFAULT_IMAGE_TOKEN}</Image>"
        for i, (role, message) in enumerate(messages):
            prompt += f"<|start_header_id|>{role}<|end_header_id|>"
            if message:
                prompt += f"{message}<|eot_id|>"
        return prompt

    def get_prompt(self):
        messages = self.messages

        if self.sep_style == SeparatorStyle.PLAIN:
            return self.parse_plain(messages)

        elif self.sep_style == SeparatorStyle.LLAMA_3:
            return self.parse_llama_3(messages)

        raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=self.messages.copy(),
            offset=self.offset,
            sep_style=self.sep_style,
            multimodal=self.multimodal,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
)

conv_llama_3 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language. The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("user", "assistant"),
    multimodal=True,
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
)


default_conversation = conv_llava_plain
conv_templates = {
    "plain": conv_llava_plain,
    "llama_3": conv_llama_3,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
