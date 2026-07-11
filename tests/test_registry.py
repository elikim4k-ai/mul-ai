from mul.config import PoolEntry, Settings


def test_disabled_entries_not_registered(registry):
    assert all(w.entry.provider != "google" for w in registry.workers)


def test_capability_resolves_to_highest_rank(registry):
    w = registry.resolve("code")
    assert w.entry.provider == "openai"  # rank 5 beats lg (3) and vllm (1)


def test_opt_out_changes_resolution(registry):
    w = registry.resolve("code", exclude_providers={"openai"})
    assert w.entry.provider == "lg"


def test_allow_list_restricts_pool(registry):
    w = registry.resolve("code", allow_providers={"vllm"})
    assert w.entry.provider == "vllm"


def test_unknown_capability_falls_back_to_best_eligible(registry):
    w = registry.resolve("vision")  # only the disabled google entry had vision
    assert w is not None
    assert w.entry.provider == "openai"  # highest rank overall


def test_fully_excluded_pool_resolves_none(registry):
    all_providers = {w.entry.provider for w in registry.workers}
    assert registry.resolve("code", exclude_providers=all_providers) is None


def test_keyless_providers_are_stubs(registry):
    # test settings carry no API keys, so every remote provider is a mock stub
    for w in registry.workers:
        if w.entry.provider in ("upstage", "lg", "openai"):
            assert w.client.is_stub
            # the stub still reports the real provider so opt-outs keep working
            assert w.client.provider == w.entry.provider


def test_key_activates_real_adapter(config):
    settings = Settings(_env_file=None, upstage_api_key="sk-test")
    from mul.workers.registry import WorkerRegistry

    registry = WorkerRegistry(config, settings)
    upstage = next(w for w in registry.workers if w.entry.provider == "upstage")
    assert not upstage.client.is_stub
    assert type(upstage.client).__name__ == "OpenAICompatClient"
