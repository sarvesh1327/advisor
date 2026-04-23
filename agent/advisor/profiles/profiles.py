from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class AdvisorTrainingConfig(BaseModel):
    backend: str
    rollout_group_size: int
    num_generations: int
    max_steps: int
    max_prompt_tokens: int
    max_completion_tokens: int
    checkpoint_root: str
    base_model_name: str | None = None
    adapter_method: str | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float = 0.0
    target_modules: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_adapter_settings(self) -> "AdvisorTrainingConfig":
        # Keep training config permissive by default, but validate explicit LoRA settings tightly.
        if self.adapter_method is None:
            return self
        if self.adapter_method != "lora":
            raise ValueError(f"unsupported adapter_method: {self.adapter_method}")
        if self.lora_rank is None or self.lora_rank <= 0:
            raise ValueError("lora_rank must be > 0 when adapter_method='lora'")
        if not self.target_modules:
            raise ValueError("target_modules must be set when adapter_method='lora'")
        if self.lora_alpha is not None and self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be > 0 when provided")
        if not 0.0 <= self.lora_dropout <= 1.0:
            raise ValueError("lora_dropout must be between 0.0 and 1.0")
        return self


class AdvisorProfile(BaseModel):
    profile_id: str
    domain: str
    description: str | None = None
    reward_spec_id: str | None = None
    training: AdvisorTrainingConfig | None = None


class AdvisorProfileRegistry(BaseModel):
    default_profile_id: str
    profiles: dict[str, AdvisorProfile] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_default_profile(self) -> "AdvisorProfileRegistry":
        if self.default_profile_id not in self.profiles:
            raise ValueError(f"default_profile_id must reference a registered profile: {self.default_profile_id}")
        return self

    @classmethod
    def from_toml(cls, path: str | Path) -> "AdvisorProfileRegistry":
        config_path = Path(path).expanduser()
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        profiles = {
            profile_id: AdvisorProfile(profile_id=profile_id, **payload)
            for profile_id, payload in (raw.get("profiles") or {}).items()
        }
        return cls(default_profile_id=raw["default_profile_id"], profiles=profiles)

    def get(self, profile_id: str) -> AdvisorProfile:
        try:
            return self.profiles[profile_id]
        except KeyError as exc:
            raise ValueError(f"unknown advisor profile: {profile_id}") from exc

    def resolve(self, profile_id: str | None = None) -> AdvisorProfile:
        return self.get(profile_id or self.default_profile_id)
