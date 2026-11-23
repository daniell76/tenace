# Hill-Climb Iter 2: Parse Bridge Card Labels in `SeatConstraint.likely_cards`

**Goal:** Fix live production `ValidationError` discovered on both parallel matches (`seed=100` and `seed=200`) where `gemini-4-argon` naturally returned bridge card names in `SeatConstraint.likely_cards` (`'D:K'`, `'DK'`, `'H:A'`, `'C:T'`, `'SK'`) instead of raw OpenSpiel integers (`0..51`).

## Changes
1. Add `_coerce_card_id` helper inside `src/tenace/domain/types.py` called by `SeatConstraint._normalize_enum_names` to parse canonical card notation into OpenSpiel integer `CardId` (`0..51`):
   - `S:A`, `C:2`, `D:K`, `C:T`, `D:10` (suit-first with colon)
   - `DK`, `SK`, `HT`, `D10` (suit-first)
   - `KD`, `AS`, `10D` (rank-first)
   - Unicode suit symbols (`♠A`, `♥K`, `♦Q`, `♣2`)
   - Preserves strict rejection of out-of-range integers (`-1`, `52`) and unknown tokens (`"invalid"`).
   - Also tolerates `likely_cards: null` as `[]` when an LLM emits JSON null for an optional list, and adds one retry guard hint in `StrategistEngine` system prompt telling Layer 1 to use `"S:A"` notation (or `[]`).
2. Strictly TDD: write failing tests in `tests/test_domain_types.py` first (`RED`), implement (`GREEN`), run tests, commit, and re-run `seed=100` and `seed=200` vs BEN.
