import dataclasses
from enum import auto, Enum
from typing import List


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

    def parse_plain(self, messages):
        chunks = []
        for i, (role, message) in enumerate(messages):
            if message:
                chunks.append((message + "\n", role == "gpt"))
        return chunks

    def parse_llama_3(self, messagse):
        chunks = []
        chunks.append(
            (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.system}<|eot_id|>",
                False,
            )
        )
        for i, (role, message) in enumerate(messages):
            chunks.append(
                (f"<|start_header_id|>{self.roles[role]}<|end_header_id|>", False)
            )
            chunks.append((f"{message}<|eot_id|>", self.roles == "gpt"))
        return chunks

    def get_prompt(self):
        messages = self.messages
        assert (
            messages[0][0] == "human"
        ), f"First message must be from human. received: {messages}"

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
    system="A chat between a curious user and an artificial intelligence assistant. The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language. The visual content will be provided with the following format: {DEFAULT_IM_START_TOKEN}visual content{DEFAULT_IM_END_TOKEN}.",
    roles=("user", "assistant"),
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
