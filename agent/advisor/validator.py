from __future__ import annotations

from .schemas import AdviceBlock, RelevantFile, RelevantSymbol


class AdviceValidator:
    def __init__(self, *, max_items: int = 8, max_plan_steps: int = 8):
        self.max_items = max_items
        self.max_plan_steps = max_plan_steps

    def validate(self, advice: AdviceBlock) -> AdviceBlock:
        # Validation trims output into a bounded, storage-safe shape without changing high-level intent.
        confidence = min(max(advice.confidence, 0.0), 1.0)
        return AdviceBlock(
            task_type=advice.task_type,
            relevant_files=self._dedupe_files(advice.relevant_files)[: self.max_items],
            relevant_symbols=self._dedupe_symbols(advice.relevant_symbols)[: self.max_items],
            constraints=self._trim(advice.constraints, self.max_items),
            likely_failure_modes=self._trim(advice.likely_failure_modes, self.max_items),
            recommended_plan=self._trim(advice.recommended_plan, self.max_plan_steps),
            avoid=self._trim(advice.avoid, self.max_items),
            confidence=confidence,
            notes=(advice.notes or "")[:500] or None,
        )

    def _trim(self, items: list[str], limit: int) -> list[str]:
        deduped = []
        seen = set()
        for item in items:
            text = (item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text[:240])
        return deduped[:limit]

    def _dedupe_files(self, items: list[RelevantFile]) -> list[RelevantFile]:
        out = []
        seen = set()
        for item in items:
            if item.path in seen:
                continue
            seen.add(item.path)
            out.append(RelevantFile(path=item.path, why=item.why[:240], priority=item.priority))
        return out

    def _dedupe_symbols(self, items: list[RelevantSymbol]) -> list[RelevantSymbol]:
        out = []
        seen = set()
        for item in items:
            key = (item.name, item.path)
            if key in seen:
                continue
            seen.add(key)
            out.append(RelevantSymbol(name=item.name[:120], path=item.path, why=item.why[:240]))
        return out
