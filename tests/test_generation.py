import pytest
from src.generation.prompts import PromptManager


def test_prompt_manager_local():
    pm = PromptManager()
    system = pm.get_prompt("system")
    assert "helpful assistant" in system.lower()

    response = pm.get_prompt("response")
    assert "{context_text}" in response
    assert "{user_question}" in response


def test_prompt_manager_unknown():
    pm = PromptManager()
    with pytest.raises(KeyError):
        pm.get_prompt("nonexistent_prompt")


def test_prompt_manager_all_prompts():
    pm = PromptManager()
    for name in ["system", "response", "summarize", "image_describe", "hyde", "multi_query", "step_back"]:
        prompt = pm.get_prompt(name)
        assert prompt, f"Prompt '{name}' returned empty"
