"""Shared model configuration for ALL agents (A, A2, B).

One configuration, reused by every agent, so the A-vs-A2-vs-B comparison holds
model/provider/decoding constant and only the prompt/candidate-budget differs.

Model/decoding choices (documented, fixed):
  * model         = "qwen3:4b"   (local Ollama; verified installed, exact tag)
  * provider      = Ollama local HTTP API at http://localhost:11434/api/generate
  * temperature   = 0.0          deterministic/reproducible; agents that emit up
                                  to 2 candidates do so in one structured response
                                  rather than via sampling, so temperature stays
                                  fixed and shared.
  * top_p         = 1.0
  * num_predict   = 2048         max generation tokens. qwen3:4b reasons innately
                                  and neither /no_think nor think:false stops it in
                                  the installed Ollama build; a small budget gets
                                  truncated mid-reasoning (empty answer). 2048 lets
                                  it finish reasoning and emit the SQL.
  * think         = True         We do NOT request chain-of-thought. qwen3 reasons
                                  on its own; with think=True Ollama isolates that
                                  reasoning into a separate `thinking` field and
                                  puts the clean SQL in `response`. The agent reads
                                  ONLY `response`, discards `thinking`, and never
                                  exposes the reasoning transcript in its candidate.
  * request_timeout = 600.0 s    configurable. CPU inference of qwen3:4b is slow
                                  (~100-200 s/query incl. reasoning); large default.

generation_settings() returns the exact knobs recorded in every AgentOutput,
plus the client-side request_timeout, for reproducibility. Note that Ollama's
eval_count (recorded by the agent as total_generated_tokens) counts ALL
generated tokens, including the discarded `thinking` tokens -- it is not an
answer-only count. ollama_options() is the subset actually sent to Ollama.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    model: str = "qwen3:4b"
    provider: str = "ollama"
    host: str = "http://localhost:11434"
    endpoint: str = "/api/generate"
    temperature: float = 0.0
    top_p: float = 1.0
    num_predict: int = 2048
    think: bool = True
    request_timeout: float = 600.0

    @property
    def url(self) -> str:
        return self.host.rstrip("/") + self.endpoint

    def generation_settings(self) -> dict:
        """Full reproducibility record stamped into every AgentOutput: model tag,
        decoding knobs, think flag, and the client-side request timeout. This is a
        superset of ollama_options(); request_timeout is NOT an Ollama option, it
        is the urllib client timeout and is recorded here as execution metadata."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
            "think": self.think,
            "request_timeout": self.request_timeout,
        }

    def ollama_options(self) -> dict:
        """Options block for the Ollama /api/generate request."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
        }


# The single shared instance used by A / A2 / B.
DEFAULT_CONFIG = ModelConfig()
