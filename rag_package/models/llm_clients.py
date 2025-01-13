from abc import ABC, abstractmethod
import os
from anthropic import Client as AnthropicClient
from openai import OpenAI


class BaseLLMClient(ABC):
    """Abstract base class defining the interface for all LLM clients."""

    def __init__(self, model_name: str, temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self._client = None

    @abstractmethod
    def get_client(self) -> any:
        """Initialize and return the API client."""
        pass

    @abstractmethod
    def create_message(self, prompt: str, max_tokens: int | None = None) -> dict:
        """Create a message using the LLM."""
        pass


class AnthropicClient(BaseLLMClient):
    """Client implementation for Anthropic's Claude models."""

    def get_client(self) -> AnthropicClient:
        if self._client is None:
            self._client = AnthropicClient()
        return self._client

    def create_message(self, prompt: str, max_tokens: int | None = None) -> dict:
        client = self.get_client()
        try:
            response = client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=max_tokens or 4096,
                temperature=self.temperature,
            )

            return {
                "content": response.content[0].text if isinstance(response.content, list) else str(response.content),
                "model": self.model_name,
                "provider": "anthropic"
            }
        except Exception as e:
            raise Exception(f"Anthropic API call failed: {str(e)}")


class OpenAIChatClient(BaseLLMClient):
    """Client implementation for OpenAI's chat-based models (GPT-3.5-turbo, GPT-4)."""

    def get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def create_message(self, prompt: str, max_tokens: int | None = None) -> dict:
        """
        Creates a message using OpenAI's chat completions API.

        This method is specifically for models that use the chat interface, like GPT-3.5-turbo
        and GPT-4. The raw OpenAI client is accessed to make the API call, while our class
        provides a consistent interface for the application.
        """
        # Get the underlying OpenAI client instance
        openai_client = self.get_client()

        try:
            # Make the API call using the raw OpenAI client
            completion = openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=max_tokens or 4096,
                temperature=self.temperature
            )

            # Format the response in our standard structure
            return {
                "content": completion.choices[0].message.content,
                "model": self.model_name,
                "provider": "openai"
            }
        except Exception as e:
            error_msg = f"OpenAI Chat API call failed: {str(e)}"
            print(f"Error details: {error_msg}")  # Helpful for debugging
            raise Exception(error_msg)


class OpenAIInstructClient(BaseLLMClient):
    """Client implementation for OpenAI's instruction-following models (GPT-3.5-turbo-instruct)."""

    def get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def create_message(self, prompt: str, max_tokens: int | None = None) -> dict:
        """
        Creates a message using OpenAI's completions API.

        This method is specifically for instruction-following models like GPT-3.5-turbo-instruct
        that use the completions API rather than the chat API.
        """
        # Get the underlying OpenAI client instance
        openai_client = self.get_client()

        try:
            # Make the API call using the raw OpenAI client
            completion = openai_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens or 4096,
                temperature=self.temperature
            )

            # Format the response in our standard structure
            return {
                "content": completion.choices[0].text.strip(),
                "model": self.model_name,
                "provider": "openai"
            }
        except Exception as e:
            error_msg = f"OpenAI Completions API call failed: {str(e)}"
            print(f"Error details: {error_msg}")  # Helpful for debugging
            raise Exception(error_msg)


def create_llm_client(model_choice: str, temperature: float = 0.1) -> BaseLLMClient:
    """
    Creates the appropriate LLM client based on the model choice.

    This factory function matches each model name to its corresponding client implementation
    and API endpoint. It ensures that each model uses the correct API interface.
    """
    model_mapping = {
        # Anthropic models
        "haiku": ("claude-3-5-haiku-latest", AnthropicClient),
        "sonnet": ("claude-3-5-sonnet-latest", AnthropicClient),

        # OpenAI Chat models
        "gpt-4": ("gpt-4-turbo-preview", OpenAIChatClient),
        "gpt-3.5-turbo": ("gpt-3.5-turbo-0125", OpenAIChatClient),

        # OpenAI Instruct model
        "gpt-3.5-turbo-instruct": ("gpt-3.5-turbo-instruct", OpenAIInstructClient),
    }

    if model_choice not in model_mapping:
        raise ValueError(
            f"Unsupported model choice: {model_choice}. "
            f"Available models: {', '.join(model_mapping.keys())}"
        )

    model_name, client_class = model_mapping[model_choice]
    return client_class(model_name, temperature)