from __future__ import annotations

import json
import re
from hashlib import sha256

from .schemas import AdviceBlock
from .settings import AdvisorSettings

try:
    from mlx_lm import generate as mlx_lm_generate
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.sample_utils import make_sampler as mlx_make_sampler
except ImportError:
    mlx_lm_generate = None
    mlx_lm_load = None
    mlx_make_sampler = None


class MLXAdvisorRuntime:
    def __init__(self, settings: AdvisorSettings):
        self.settings = settings
        self._model = None
        self._tokenizer = None

    @property
    def prompt_hash_seed(self) -> str:
        return sha256(f"{self.settings.model_name}:{self.settings.model_version}".encode()).hexdigest()

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        if mlx_lm_load is None:
            raise RuntimeError("mlx-lm is not installed. Install the advisor runtime extras to enable MLX inference.")

        self._model, self._tokenizer = mlx_lm_load(self.settings.model_name)
        return self._model, self._tokenizer

    def generate_advice(self, packet) -> AdviceBlock:
        model, tokenizer = self._ensure_loaded()
        if mlx_lm_generate is None or mlx_make_sampler is None:
            raise RuntimeError("mlx-lm is not installed. Install the advisor runtime extras to enable MLX inference.")

        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a code-execution advisor. Return ONLY valid JSON with keys "
                        "task_type, relevant_files, relevant_symbols, constraints, likely_failure_modes, "
                        "recommended_plan, avoid, confidence, notes."
                    ),
                },
                {"role": "user", "content": self._format_prompt(packet)},
                {"role": "assistant", "content": "{"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        response = mlx_lm_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self.settings.max_tokens,
            sampler=mlx_make_sampler(temp=self.settings.temperature),
            verbose=False,
        )
        payload = self._coerce_payload(self._extract_json(response))
        return AdviceBlock.model_validate(payload)

    def _format_prompt(self, packet) -> str:
        return (
            "You are a code-execution advisor that emits JSON only.\n"
            "Return an object with keys: task_type, relevant_files, relevant_symbols, constraints, likely_failure_modes, recommended_plan, avoid, confidence, notes.\n"
            "Keep output concise and useful.\n\n"
            f"TASK: {packet.task_text}\n"
            f"TASK_TYPE: {packet.task_type}\n"
            f"REPO: {packet.repo}\n"
            f"MODULES: {packet.repo_summary.modules}\n"
            f"HOTSPOTS: {packet.repo_summary.hotspots}\n"
            f"FILE_TREE: {packet.repo_summary.file_tree_slice}\n"
            f"CANDIDATE_FILES: {[item.model_dump() for item in packet.candidate_files]}\n"
            f"RECENT_FAILURES: {[item.model_dump() for item in packet.recent_failures]}\n"
            f"CONSTRAINTS: {packet.constraints}\n"
            f"ACCEPTANCE_CRITERIA: {packet.acceptance_criteria}\n"
        )

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        fenced = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(text)
        return payload

    def _coerce_payload(self, payload: dict) -> dict:
        def _to_strings(items):
            if isinstance(items, str):
                items = [items]
            out = []
            for item in items or []:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    action = item.get("action") or item.get("name") or item.get("path") or json.dumps(item, sort_keys=True)
                    file_part = item.get("file") or item.get("path")
                    out.append(f"{action} ({file_part})" if file_part and file_part not in action else str(action))
                else:
                    out.append(str(item))
            return out

        relevant_files = []
        for idx, item in enumerate(payload.get("relevant_files") or [], start=1):
            if isinstance(item, str):
                relevant_files.append({"path": item, "why": "advisor-selected relevant file", "priority": idx})
            elif isinstance(item, dict) and item.get("path"):
                relevant_files.append({
                    "path": item["path"],
                    "why": item.get("why") or item.get("reason") or "advisor-selected relevant file",
                    "priority": int(item.get("priority") or idx),
                })

        relevant_symbols = []
        for item in payload.get("relevant_symbols") or []:
            if isinstance(item, str):
                relevant_symbols.append({"name": item, "path": "", "why": "advisor-selected symbol"})
            elif isinstance(item, dict):
                relevant_symbols.append({
                    "name": item.get("name") or item.get("symbol") or "unknown",
                    "path": item.get("path") or "",
                    "why": item.get("why") or item.get("reason") or "advisor-selected symbol",
                })

        return {
            "task_type": payload.get("task_type") or "feature",
            "relevant_files": relevant_files,
            "relevant_symbols": relevant_symbols,
            "constraints": _to_strings(payload.get("constraints") or []),
            "likely_failure_modes": _to_strings(payload.get("likely_failure_modes") or []),
            "recommended_plan": _to_strings(payload.get("recommended_plan") or []),
            "avoid": _to_strings(payload.get("avoid") or []),
            "confidence": float(payload.get("confidence") or 0.0),
            "notes": str(payload.get("notes") or "") or None,
        }
