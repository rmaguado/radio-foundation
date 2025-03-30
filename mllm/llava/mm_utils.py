import torch

from mllm.llava.constants import IMAGE_TOKEN_INDEX


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def detokenizer_image_token(token_ids, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    """
    Converts a sequence of token IDs back to a string, replacing image tokens with a placeholder.

    Args:
        token_ids (list[int]): List of token IDs.
        tokenizer: Tokenizer with a decode method.
        image_token_index (int): The token ID representing an image.

    Returns:
        str: The reconstructed text with image placeholders.
    """
    text_chunks = []
    current_chunk = []

    for token in token_ids:
        if token == image_token_index:
            if current_chunk:
                text_chunks.append(tokenizer.decode(current_chunk))
                current_chunk = []
            text_chunks.append("<image>")
        else:
            current_chunk.append(token)

    if current_chunk:
        text_chunks.append(tokenizer.decode(current_chunk))

    return "".join(text_chunks).strip()
