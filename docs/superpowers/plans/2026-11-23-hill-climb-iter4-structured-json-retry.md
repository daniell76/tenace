# Hill-Climb Iter 4: Structured JSON Retry & `null` Optional-Field Tolerance

**Goal:** Prevent a single malformed/truncated `gemini-4-argon` response (caught live on `seed=400` after ~8 min: `ValueError: No valid JSON object found in model response`) from aborting a multi-board match or causing HTTP `502` on Cloud Run.

## Changes
1. **`VertexGeminiClient.generate_structured` (`src/tenace/models/clients.py`):**
   - Add `max_parse_retries: int = 2` parameter (validates `>= 0`).
   - On empty text (`ValueError`), malformed/truncated JSON (`ValueError`), or Pydantic schema failure (`ValidationError`), retry up to `max_parse_retries` times appending a brief diagnostic reminder citing the parse error. Re-raise the final exception if all attempts fail.
2. **`StrategyState` null optional-field coercion (`src/tenace/domain/types.py`):**
   - Coerce explicit JSON `null` on optional fields (`target_or_threat: null -> ""`, `assumptions: null -> []`, `seat_constraints: null -> {}`).
3. **TDD tests (`tests/test_model_clients.py`, `tests/test_domain_types.py`):**
   - Verify `VertexGeminiClient` recovers on 2nd attempt after malformed JSON / `ValidationError` / empty text and includes feedback in retry prompt.
   - Verify `VertexGeminiClient` still raises the original `ValueError` / `ValidationError` when all retries fail or `max_parse_retries=0`.
   - Verify `StrategyState` handles `null` optional fields gracefully.
