import llm


def test_plugin_is_installed():
    model_ids = [model.model_id for model in llm.get_models()]
    assert "openai/gpt-4o-mini" in model_ids
