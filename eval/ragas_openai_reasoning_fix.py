"""
eval/ragas_openai_reasoning_fix.py
------------------------------------
A real, measured bug in ragas==0.4.3, not a guess: `InstructorLLM._map_openai_params()`
(ragas/llms/base.py) auto-detects "reasoning" models (o-series, GPT-5+) that require
`max_completion_tokens` instead of `max_tokens` (and temperature pinned to 1.0, no top_p)
via a regex-free version parse:

    version_str = model_str[4:].split("-")[0].split("_")[0]
    version = int(version_str)   # <-- raises ValueError for a minor-version id

For a model id with a decimal minor version -- like "gpt-5.6-luna" -- `version_str` is
"5.6", and `int("5.6")` raises ValueError, which the library silently catches and treats
as "not a reasoning model". So the rename/param-fix never happens, and the real OpenAI API
(correctly) rejects the resulting `max_tokens=1024` with:
    "Unsupported parameter: 'max_tokens' is not supported with this model.
     Use 'max_completion_tokens' instead."
This was confirmed against ragas 0.4.3's actual installed source (not assumed): see
DECISIONS.md.

There is no public llm_factory()/create_llm() parameter that lets a caller supply a
pre-built InstructorModelArgs or otherwise skip the buggy auto-detection -- create_llm()
hardcodes `model_args=InstructorModelArgs()` and merges any extra kwargs on top, so the
default `max_tokens: 1024` field is always present regardless of what we pass in.

THE FIX: `_map_openai_params()` returns `self.model_args.copy()` completely unmodified
whenever its (buggy) detection says "not a reasoning model" -- so if we set
`llm.model_args` directly, right after construction, to the EXACT shape the library's own
(correct, but unreachable for this model id) reasoning-model branch would have produced,
every subsequent call sends exactly the right request. Verified directly against the
installed ragas==0.4.3 package (constructing a real InstructorLLM and calling
`_map_openai_params()` before/after this patch) -- not assumed. This is an instance-level
data fix (no monkeypatching of ragas's classes or methods), scoped to the one LLM instance
this project's judge calls use.
"""


def patch_reasoning_model_args(llm, max_completion_tokens: int = 4096):
    """Mutates an `InstructorLLM` (as returned by `ragas.llms.base.llm_factory`) in place
    so its OpenAI-bound calls use `max_completion_tokens` (not `max_tokens`), temperature
    pinned to 1.0, and no `top_p` -- the same three changes ragas's own reasoning-model
    branch applies, for models its own detection can actually recognize. Call this once,
    immediately after `llm_factory(...)`, before any `.ascore()`/`.agenerate()` call.

    `max_completion_tokens` defaults to 4096, not the library's own default of 1024:
    InstructorModelArgs's own docstring warns that for GPT-5-and-newer models scored via
    structured (Pydantic) output, 1024 "may not be sufficient" and recommends 4096+."""
    llm.model_args.pop("max_tokens", None)
    llm.model_args["max_completion_tokens"] = max_completion_tokens
    llm.model_args["temperature"] = 1.0
    llm.model_args.pop("top_p", None)
    return llm
