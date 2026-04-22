from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from hashlib import sha256

from .schemas import AdviceBlock
from .settings import AdvisorSettings

try:
    # MLX stays optional so the module can still import in fallback-only or test environments.
    from mlx_lm import generate as mlx_lm_generate
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.sample_utils import make_sampler as mlx_make_sampler
except ImportError:
    mlx_lm_generate = None
    mlx_lm_load = None
    mlx_make_sampler = None

_MLX_MISSING_REASON = (
    "mlx-lm is not installed. Install the advisor runtime extras to enable MLX inference."
)


class MLXAdvisorRuntime:
    def __init__(self, settings: AdvisorSettings):
        self.settings = settings
        self._model = None
        self._tokenizer = None
        self._active_model_name = settings.model_name

    @property
    def prompt_hash_seed(self) -> str:
        seed = f"{self._active_model_name}:{self.settings.model_version}"
        return sha256(seed.encode()).hexdigest()

    def capabilities(self) -> dict:
        # Availability means "can answer somehow"; readiness means the primary runtime is already loaded.
        available = (
            mlx_lm_load is not None
            and mlx_lm_generate is not None
            and mlx_make_sampler is not None
        ) or self.settings.enable_fallback_runtime
        ready = (
            (self._model is not None and self._tokenizer is not None)
            or self.settings.enable_fallback_runtime
        )
        reason = None
        if not available:
            reason = _MLX_MISSING_REASON
        elif self._model is None and self._tokenizer is None and self.settings.enable_fallback_runtime:
            reason = "heuristic fallback runtime available"
        return {
            "runtime": "mlx",
            "available": available,
            "ready": ready,
            "reason": reason,
            "model_name": self.settings.model_name,
            "active_model_name": self._active_model_name,
            "model_version": self.settings.model_version,
        }

    def warmup(self) -> None:
        try:
            self._ensure_loaded()
        except RuntimeError:
            if not self.settings.enable_fallback_runtime:
                raise

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        if mlx_lm_load is None:
            raise RuntimeError(_MLX_MISSING_REASON)

        try:
            self._model, self._tokenizer = mlx_lm_load(self.settings.model_name)
            self._active_model_name = self.settings.model_name
        except Exception:
            # Fallback model loading is only for degraded continuity, not silent model switching.
            if not self.settings.fallback_model_name:
                raise
            self._model, self._tokenizer = mlx_lm_load(self.settings.fallback_model_name)
            self._active_model_name = self.settings.fallback_model_name
        return self._model, self._tokenizer

    def generate_advice(self, packet, system_prompt: str | None = None) -> AdviceBlock:
        try:
            model, tokenizer = self._ensure_loaded()
            if mlx_lm_generate is None or mlx_make_sampler is None:
                raise RuntimeError(_MLX_MISSING_REASON)
        except Exception as exc:
            if self.settings.enable_fallback_runtime:
                return self._heuristic_fallback(packet, reason=str(exc))
            raise

        prompt = self._build_generation_prompt(tokenizer, packet, system_prompt=system_prompt)

        last_error: Exception | None = None
        for _attempt in range(self.settings.max_retries + 1):
            try:
                response = self._generate_response(model, tokenizer, prompt)
                payload = self._coerce_payload(self._extract_json(response))
                return AdviceBlock.model_validate(payload)
            except (json.JSONDecodeError, TimeoutError, ValueError) as exc:
                # Retry only malformed or timeout-style failures; other exceptions escape earlier.
                last_error = exc

        if self.settings.enable_fallback_runtime:
            reason = str(last_error) if last_error else "generation failed"
            return self._heuristic_fallback(packet, reason=reason)
        if last_error:
            raise last_error
        raise RuntimeError("generation failed without a recoverable error")

    def _build_generation_prompt(self, tokenizer, packet, system_prompt: str | None = None) -> str:
        effective_system_prompt = system_prompt or self.settings.system_prompt
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": effective_system_prompt,
                },
                {"role": "user", "content": self._format_prompt(packet)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate_response(self, model, tokenizer, prompt: str) -> str:
        if mlx_lm_generate is None or mlx_make_sampler is None:
            raise RuntimeError(_MLX_MISSING_REASON)

        def _run_generate():
            return mlx_lm_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.settings.max_tokens,
                sampler=mlx_make_sampler(temp=self.settings.temperature),
                verbose=False,
            )

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_generate)
        try:
            result = future.result(timeout=self.settings.inference_timeout_seconds)
        except FutureTimeoutError as exc:
            # Do not wait on shutdown here or the timeout path defeats its own latency bound.
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError("generation exceeded timeout") from exc
        except Exception:
            executor.shutdown(wait=True, cancel_futures=False)
            raise

        executor.shutdown(wait=True, cancel_futures=False)
        return result

    def _heuristic_fallback(self, packet, *, reason: str) -> AdviceBlock:
        # Keep fallback advice deterministic and minimal so it is clearly weaker than model output.
        top_file = packet.candidate_files[0].path if packet.candidate_files else None
        recommended_plan = []
        if top_file:
            recommended_plan.append(f"inspect {top_file}")
        recommended_plan.append("run a focused verification step")
        relevant_files = []
        if top_file:
            relevant_files.append({"path": top_file, "why": "top candidate from repo scan", "priority": 1})
        return AdviceBlock.model_validate(
            {
                "task_type": packet.task_type,
                "focus_targets": [
                    {
                        "kind": "file",
                        "locator": top_file,
                        "rationale": "top candidate from repo scan",
                        "priority": 1,
                    }
                ] if top_file else [],
                "relevant_files": relevant_files,
                "relevant_symbols": [],
                "constraints": packet.constraints,
                "likely_failure_modes": [reason],
                "recommended_plan": recommended_plan,
                "avoid": ["broad refactors before confirming the failing area"],
                "confidence": 0.35,
                "notes": f"heuristic fallback runtime used: {reason}",
            }
        )

    def _format_prompt(self, packet) -> str:
        # Keep the prompt generic-first so adapters only add domain-specific cues through packet data.
        focus_targets = [
            {
                "kind": item.kind,
                "locator": item.locator,
                "rationale": item.description,
                "priority": index + 1,
            }
            for index, item in enumerate(packet.artifacts)
        ]
        json_template = (
            '{"task_type": "execution", "focus_targets": '
            '[{"kind": "document", "locator": "docs/brief.md", "rationale": "task brief", "priority": 1}], '
            '"relevant_files": [], "relevant_symbols": [], "constraints": [], '
            '"likely_failure_modes": ["acting on stale context"], "recommended_plan": '
            '["review the brief", "run a focused verification step"], "avoid": ["broad unverified changes"], '
            '"confidence": 0.8, "notes": "brief note", '
            '"injection_policy": {"strategy": "prepend", "format": "plain_text", '
            '"min_confidence": 0.0, "include_confidence_note": true}}'
        )
        return (
            "You are an execution advisor that emits JSON only.\n"
            "Return exactly one JSON object with keys: task_type, focus_targets, "
            "relevant_files, relevant_symbols, constraints, likely_failure_modes, "
            "recommended_plan, avoid, confidence, notes, injection_policy.\n"
            "Rules:\n"
            "- Output must start with { and end with }\n"
            "- Do not include markdown fences\n"
            "- Do not include role tags or prose before/after the JSON\n"
            "- Keep output concise and useful\n\n"
            f"TASK: {packet.task_text}\n"
            f"TASK_TYPE: {packet.task_type}\n"
            f"TASK_DOMAIN: {packet.task.domain}\n"
            f"CONTEXT_SUMMARY: {packet.context.summary}\n"
            f"REPO: {packet.repo}\n"
            f"MODULES: {packet.repo_summary.modules}\n"
            f"HOTSPOTS: {packet.repo_summary.hotspots}\n"
            f"FILE_TREE: {packet.repo_summary.file_tree_slice}\n"
            f"CANDIDATE_FILES: {[item.model_dump() for item in packet.candidate_files]}\n"
            f"ARTIFACTS: {[item.model_dump() for item in packet.artifacts]}\n"
            f"RECENT_FAILURES: {[item.model_dump() for item in packet.recent_failures]}\n"
            f"HISTORY: {[item.model_dump() for item in packet.history]}\n"
            f"CONSTRAINTS: {packet.constraints}\n"
            f"ACCEPTANCE_CRITERIA: {packet.acceptance_criteria}\n"
            f"DOMAIN_CAPABILITIES: {[item.model_dump() for item in packet.domain_capabilities]}\n"
            f"FOCUS_TARGETS: {focus_targets}\n"
            f"JSON_TEMPLATE: {json_template}\n"
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
            # Normalize loose model output into the strict schema without discarding useful intent.
            if isinstance(items, str):
                items = [items]
            out = []
            for item in items or []:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    action = (
                        item.get("action")
                        or item.get("name")
                        or item.get("path")
                        or json.dumps(item, sort_keys=True)
                    )
                    file_part = item.get("file") or item.get("path")
                    if file_part and file_part not in action:
                        out.append(f"{action} ({file_part})")
                    else:
                        out.append(str(action))
                else:
                    out.append(str(item))
            return out

        focus_targets = []
        for idx, item in enumerate(payload.get("focus_targets") or [], start=1):
            if isinstance(item, str):
                focus_targets.append(
                    {
                        "kind": "artifact",
                        "locator": item,
                        "rationale": "advisor-selected focus target",
                        "priority": idx,
                    }
                )
            elif isinstance(item, dict) and item.get("locator"):
                focus_targets.append(
                    {
                        "kind": str(item.get("kind") or "artifact"),
                        "locator": item["locator"],
                        "rationale": item.get("rationale") or item.get("why") or item.get("reason") or "advisor-selected focus target",
                        "priority": int(item.get("priority") or idx),
                    }
                )

        relevant_files = []
        for idx, item in enumerate(payload.get("relevant_files") or [], start=1):
            if isinstance(item, str):
                relevant_files.append(
                    {
                        "path": item,
                        "why": "advisor-selected relevant file",
                        "priority": idx,
                    }
                )
            elif isinstance(item, dict) and item.get("path"):
                relevant_files.append(
                    {
                        "path": item["path"],
                        "why": item.get("why")
                        or item.get("reason")
                        or "advisor-selected relevant file",
                        "priority": int(item.get("priority") or idx),
                    }
                )

        relevant_symbols = []
        for item in payload.get("relevant_symbols") or []:
            if isinstance(item, str):
                relevant_symbols.append(
                    {"name": item, "path": "", "why": "advisor-selected symbol"}
                )
            elif isinstance(item, dict):
                relevant_symbols.append(
                    {
                        "name": item.get("name") or item.get("symbol") or "unknown",
                        "path": item.get("path") or "",
                        "why": item.get("why")
                        or item.get("reason")
                        or "advisor-selected symbol",
                    }
                )

        coerced_payload = {
            "task_type": payload.get("task_type") or "feature",
            "focus_targets": focus_targets,
            "relevant_files": relevant_files,
            "relevant_symbols": relevant_symbols,
            "constraints": _to_strings(payload.get("constraints") or []),
            "likely_failure_modes": _to_strings(
                payload.get("likely_failure_modes") or []
            ),
            "recommended_plan": _to_strings(payload.get("recommended_plan") or []),
            "avoid": _to_strings(payload.get("avoid") or []),
            "confidence": float(payload.get("confidence") or 0.0),
            "notes": str(payload.get("notes") or "") or None,
        }
        if payload.get("injection_policy") is not None:
            coerced_payload["injection_policy"] = payload["injection_policy"]
        return coerced_payload
