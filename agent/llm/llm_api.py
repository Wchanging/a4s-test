import openai
from typing import Any, Dict, List, Optional


def call_openai_api(model: str,
                    messages: List[Dict[str, Any]],
                    temperature: float = 0.3,
                    max_tokens: int = 150,
                    top_p: float = 1.0,
                    frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0,
                    stop: Optional[List[str]] = None
                    ) -> Dict[str, Any]:
    """
    Call the OpenAI API to generate a response based on the provided messages.

    :param model: The model to use for generating the response.
    :param messages: A list of message dictionaries containing role and content.
    :param temperature: Sampling temperature.
    :param max_tokens: Maximum number of tokens to generate.
    :param top_p: Nucleus sampling parameter.
    :param frequency_penalty: Frequency penalty for generated tokens.
    :param presence_penalty: Presence penalty for generated tokens.
    :param stop: Optional list of stop sequences.
    :return: The API response as a dictionary.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )

    return response
