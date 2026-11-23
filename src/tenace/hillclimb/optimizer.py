from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from tenace.hillclimb.models import (
    BoardTelemetryRecord,
    PromptVariables,
    ProposedPromptPatch,
    SkillVersionMeta,
)
from tenace.models.clients import StructuredModelClient

__all__ = [
    "GeminiSkillOptimizer",
    "ManualPatchProposer",
    "PromptProposer",
]


@runtime_checkable
class PromptProposer(Protocol):
    """Interface for proposing the next candidate bidding-system prompt patch."""

    name: str

    def propose(
        self,
        *,
        system_id: str,
        current_version: str,
        current_variables: PromptVariables,
        losing_boards: Sequence[BoardTelemetryRecord],
        version_history: Sequence[SkillVersionMeta],
    ) -> ProposedPromptPatch | None: ...


class ManualPatchProposer:
    """Deterministic proposer that yields a pre-supplied sequence of patches (offline/testing)."""

    def __init__(self, patches: Sequence[ProposedPromptPatch], *, name: str = "manual") -> None:
        self._patches = list(patches)
        self.name = name

    def propose(
        self,
        *,
        system_id: str,
        current_version: str,
        current_variables: PromptVariables,
        losing_boards: Sequence[BoardTelemetryRecord],
        version_history: Sequence[SkillVersionMeta],
    ) -> ProposedPromptPatch | None:
        del system_id, current_version, current_variables, losing_boards, version_history
        if not self._patches:
            return None
        return self._patches.pop(0)


class GeminiSkillOptimizer:
    """LLM-assisted proposer that studies worst-swing boards & emits a minimal `ProposedPromptPatch`."""

    name = "gemini-optimizer"

    def __init__(self, client: StructuredModelClient, *, max_boards_in_prompt: int = 6) -> None:
        self.client = client
        self.max_boards_in_prompt = max_boards_in_prompt

    def propose(
        self,
        *,
        system_id: str,
        current_version: str,
        current_variables: PromptVariables,
        losing_boards: Sequence[BoardTelemetryRecord],
        version_history: Sequence[SkillVersionMeta],
    ) -> ProposedPromptPatch | None:
        if not losing_boards:
            return None

        ordered = sorted(losing_boards, key=lambda b: b.imps_challenger)[
            : self.max_boards_in_prompt
        ]
        board_blocks: list[str] = []
        for b in ordered:
            ref_r = b.table_a if b.match_mode == "controlled" else b.table_b
            chal_r = b.table_b if b.match_mode == "controlled" else b.table_a
            div_str = (
                f"call #{b.first_divergence.call_index + 1} ({b.first_divergence.seat}): "
                f"ref={b.first_divergence.reference_call} vs chal={b.first_divergence.challenger_call}"
                if b.first_divergence is not None
                else "identical auction (card-play swing)"
            )
            hands_lines = "\n".join(
                f"    {seat}: ({b.deal.hand_hcp.get(seat, 0)} HCP, {b.deal.shapes.get(seat, '')}) {b.deal.hands.get(seat, '')}"
                for seat in ("NORTH", "EAST", "SOUTH", "WEST")
            )
            board_blocks.append(
                f"- Board {b.board_key} | Dealer={b.dealer} Vul={b.vul} | "
                f"IMP={b.imps_challenger:+d} ({b.swing_category})\n"
                f"  Ref : {ref_r.contract} (NS {ref_r.ns_score:+.0f}) | {ref_r.auction}\n"
                f"  Chal: {chal_r.contract} (NS {chal_r.ns_score:+.0f}) | {chal_r.auction}\n"
                f"  First divergence: {div_str}\n"
                f"  Hands:\n{hands_lines}"
            )

        history_lines = [
            f"- {vm.version} [{vm.status.upper()}]: {vm.hypothesis} (changed={vm.changed_slots})"
            for vm in version_history[-8:]
        ]
        system_prompt = (
            "You are Tenace's Bridge Bidding-System Calibration Optimizer.\n"
            "Given a bidding system's current tunable prompt variables and worst negative-IMP boards "
            "against a reference Bridge engine, propose ONE surgical, generalizable prompt patch.\n"
            "Rules:\n"
            "- Prefer appending 1–3 concise bullet rules to `calibration_notes_append` (leaving the "
            "core `summary_rules` canonical and uncorrupted) unless a wording in `summary_rules` is "
            "actively contradicted.\n"
            "- NEVER memorize specific card deals, board numbers, or hardcode answers to single boards.\n"
            "- NEVER contradict the bidding system's core conventions (e.g., do NOT insert straight "
            "Blackwood `5C=0/4` into an RKCB 1430 system!).\n"
            "- State a clear, falsifiable `hypothesis` citing the `target_boards` (e.g. ['3000:2']).\n"
            "- Leave any replacement field `null` if you are NOT modifying that slot."
        )
        user_prompt = (
            f"System ID: {system_id} | Active Version: {current_version}\n\n"
            f"=== Recent Version History ===\n"
            + ("\n".join(history_lines) if history_lines else "(none)")
            + "\n\n=== Current Mutable Prompt Variables ===\n"
            f"[summary_rules]\n{current_variables.summary_rules}\n\n"
            f"[detailed_rules]\n{current_variables.detailed_rules or '(empty)'}\n\n"
            f"[calibration_notes]\n{current_variables.calibration_notes or '(empty)'}\n\n"
            "=== Target Losing / Swing Boards ===\n"
            + "\n".join(board_blocks)
        )
        return self.client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=ProposedPromptPatch,
        )
