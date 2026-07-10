"""Model-free active-row and stopping ledger for merged decode-loop gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass
class ActiveRowState:
    request_id: str
    max_new_tokens: int
    eos_token_ids: tuple[int, ...] = ()
    generated_token_ids: list[int] = field(default_factory=list)
    draw_count: int = 0
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

    @property
    def active(self) -> bool:
        return self.stop_reason is None

    def record_draw(self, token_id: int) -> None:
        if not self.active:
            raise RuntimeError(f"request {self.request_id} is already stopped.")
        self.draw_count += 1
        self.generated_token_ids.append(int(token_id))
        if token_id in self.eos_token_ids:
            self.stop_reason = "eos"
        elif len(self.generated_token_ids) >= self.max_new_tokens:
            self.stop_reason = "max_new_tokens"


class ActiveRowLedger:
    """Own active/inactive request rows and forbid dummy or post-stop draws."""

    def __init__(self, rows: Sequence[ActiveRowState | None]):
        if not rows:
            raise ValueError("rows must be non-empty.")
        request_ids = [row.request_id for row in rows if row is not None]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_id values must be unique.")
        self.rows = list(rows)

    def active_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, row in enumerate(self.rows)
            if row is not None and row.active
        )

    def has_active_rows(self) -> bool:
        return bool(self.active_indices())

    def materialize_step(
            self,
            sampled_tokens: Mapping[int, int],
            *,
            pad_token_id: int,
    ) -> tuple[tuple[int, ...], tuple[bool, ...]]:
        """Record exactly one draw per active row and pad inactive/dummy slots."""

        active_before = self.active_indices()
        if set(sampled_tokens) != set(active_before):
            missing = sorted(set(active_before) - set(sampled_tokens))
            inactive = sorted(set(sampled_tokens) - set(active_before))
            raise ValueError(
                "sampled_tokens must cover active rows exactly; "
                f"missing={missing}, inactive_or_dummy={inactive}."
            )
        full_tokens: list[int] = []
        active_mask: list[bool] = []
        for index, row in enumerate(self.rows):
            is_active = index in sampled_tokens
            active_mask.append(is_active)
            if not is_active:
                full_tokens.append(int(pad_token_id))
                continue
            if row is None:
                raise AssertionError("dummy rows cannot be active.")
            token_id = int(sampled_tokens[index])
            row.record_draw(token_id)
            full_tokens.append(token_id)
        return tuple(full_tokens), tuple(active_mask)

    def report(self) -> list[dict[str, object] | None]:
        return [
            (
                {
                    "request_id": row.request_id,
                    "generated_token_ids": list(row.generated_token_ids),
                    "generated_token_count": len(row.generated_token_ids),
                    "draw_count": row.draw_count,
                    "stop_reason": row.stop_reason,
                    "active": row.active,
                }
                if row is not None
                else None
            )
            for row in self.rows
        ]
