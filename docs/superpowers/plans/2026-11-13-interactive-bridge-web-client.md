# Interactive Tenace Bridge Table & Duplicate Match Lab Web Client Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and serve a polished, zero-CORS, single-origin Contract Bridge Web Client (`GET /` + `/static/*`) directly from Tenace's Cloud Run FastAPI service so a developer can interactively play Bridge as South alongside Tenace (`North`) against Tenace or Fast Baseline (`East`/`West`), watch 4x AI deals step-by-step, inspect live Layer-1 `StrategyState` reasoning and DDS telemetry, and run two-room WBF Duplicate IMP benchmarks from the browser.

**Architecture:**
1. **Stateless Table State API (`POST /v1/table/state`) & Fast Baseline Option (`bot` field on `POST /v1/step`)**:
   - `POST /v1/table/state` accepts `(tournament_seed, board_number, action_history, viewer_seat, reveal_all)` and uses OpenSpiel (`pyspiel`) to compute ground-truth legal actions, acting seat (`current_player` vs physical `acting_hand` when declarer plays dummy's cards), WBF dealer/vulnerability, auction call history, contract summary, completed tricks + current trick on the felt, trick counts (`ns_tricks`, `ew_tricks`), HCP per visible hand, and strictly filtered card visibility (only `viewer_seat`, post-lead `dummy_seat`, or all 4 hands when `phase == "terminal"` or `reveal_all == True`; hidden hands return `null` plus exact remaining card counts).
   - `StepRequest` gains `bot: Literal["tenace", "baseline"] = "tenace"` so East/West can optionally run `FirstLegalPassBaselineBot` (`0` LLM calls) when the user wants fast solo testing with Tenace North as partner.
2. **Bundled Vanilla HTML/CSS/JS Web Application (`src/tenace/web/`)**:
   - Served at `GET /` (`HTMLResponse`) and `/static/*` (`StaticFiles`) inside `create_app()`, automatically packaged into the Python wheel (`src/tenace/web/{index.html,styles.css,app.js}`) with zero Node build requirement inside `Dockerfile`.
   - **Aesthetic & UX Direction (`web-app-development` skill)**:
     - Championship tournament baize table (`hsl(156, 44%, 15%)` radial felt with subtle woven grain), warm ivory playing cards (`hsl(42, 33%, 97%)`), editorial serif (`Fraunces`), crisp sans (`Plus Jakarta Sans`), and tabular monospace (`JetBrains Mono`), with a 1-click **4-Color / Classic** Bridge suit palette toggle (no purple-on-dark, no neon glowing borders, no gradient text, no biscuit pills).
     - **Interactive Table View**: 4-compass layout (North, West, Trick Felt Arena, East, South), authentic 7x5 WBF Bidding Box (`1C..7NT`, `Pass`, `Dbl`, `RDbl`), clickable legal cards (highlighting `acting_hand` both when South plays from hand and when South as declarer plays from North's dummy hand!), Undo / New Board / Random Seed controls, Play Mode selector (`Play as South vs Tenace E/W`, `Play as South vs Fast Baseline E/W`, `Spectator 4x Tenace`), `Auto-Play` / `Step AI` controls, and `Glass-Box Reveal Hands` toggle.
     - **AI Glass-Box Inspector & Model Selector**: Configure `mode` (`vertex` vs `mock`), `strategist_model`, `evaluator_backend`, `evaluator_model`, and `dds_worlds` (auto-initialized from `GET /health`), and inspect live Layer-1 `plan_summary`, `target_or_threat`, `assumptions`, and cumulative call telemetry per seat.
     - **Duplicate IMP Match Lab View**: Interactive runner for `POST /v1/match` (`baseline` or head-to-head `tenace` model comparison) rendering IMP swing cards, room-by-room contracts/auctions, and collapsible full text scorecard.

**Tech Stack:** Python 3.13, FastAPI (`StaticFiles`, `FileResponse`), OpenSpiel (`pyspiel`), Semantic HTML5, Vanilla CSS3, Vanilla ES2022 JavaScript.

---

### Task 1: Backend Table State Endpoint (`POST /v1/table/state`), Baseline Bot Step Option & Static Web Mount

**Files:**
- Modify: `src/tenace/server.py`
- Create: `src/tenace/web/index.html` (minimal valid shell in Task 1, full UI in Task 2)
- Create: `src/tenace/web/styles.css`
- Create: `src/tenace/web/app.js`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write failing tests in `tests/test_server.py`**
  Cover:
  - `POST /v1/table/state`:
    - Auction state: hides unseen hands (`NORTH`, `EAST`, `WEST` hands are `None` with `remaining_counts == 13` when `viewer_seat="SOUTH"` and `reveal_all=False`), exposes `SOUTH`'s 13 sorted cards + `hand_hcp`, returns `dealer`, `ns_vulnerable`, `ew_vulnerable`, `current_player`, `acting_hand`, and 38 initial `legal_actions` (`Pass`, `1C`..`7NT`).
    - Pre-lead vs post-lead dummy visibility: on opening lead (`len(play_actions) == 0`), dummy hand is still hidden (`None`); immediately after opening lead (`len(play_actions) == 1`), dummy's remaining cards become visible, `contract` metadata (`level`, `trump`, `doubled`, `declarer`, `dummy`, `formatted`) is populated, and when declarer acts on dummy's turn, `current_player` is declarer while `acting_hand` is dummy!
    - Trick tracking & terminal board: tracks `current_trick`, `completed_tricks` (`leader`, `cards`, `winner`), `ns_tricks`, `ew_tricks`, reveals all 4 initial/remaining hands on `phase == "terminal"` (and when `reveal_all=True`), and returns `terminal_result` (`ns_score`, `ew_score`, `contract`).
    - Rejects illegal `action_history` or invalid `board_number` with `422`.
  - `POST /v1/step` with `bot="baseline"`: executes `FirstLegalPassBaselineBot` in `0` strategist/evaluator calls and returns valid `StepResponse`.
  - `GET /` and `GET /static/app.js` / `GET /static/styles.css`: returns `200 OK` HTML/CSS/JS responses.
- [ ] **Step 2: Run `uv run pytest tests/test_server.py -v` to verify RED failures**
- [ ] **Step 3: Implement `POST /v1/table/state`, `StepRequest.bot`, and `GET /` + `/static` mounting in `src/tenace/server.py`**
- [ ] **Step 4: Run `uv run pytest tests/test_server.py -v` to verify GREEN**
- [ ] **Step 5: Commit Task 1** (`feat(server): add imperfect-info /v1/table/state endpoint and web static mount`)

---

### Task 2: Interactive Bridge Table & Duplicate Match Lab Frontend (`src/tenace/web/`)

**Files:**
- Modify: `src/tenace/web/index.html`
- Modify: `src/tenace/web/styles.css`
- Modify: `src/tenace/web/app.js`
- Modify: `README.md`
- Test: `tests/test_web_ui.py`

- [ ] **Step 1: Write failing frontend structure, accessibility, aesthetic guard & JS syntax tests in `tests/test_web_ui.py`**
  Cover:
  - Semantic HTML5 structure: single `<h1>`, `<title>`, `<meta name="description">`, unique descriptive IDs for table compass seats (`seat-north`, `seat-east`, `seat-south`, `seat-west`), bidding box (`bidding-box`), trick arena (`trick-arena`), strategy inspector (`strategy-feed`), model profile controls, and Duplicate Match Lab tab (`tab-play`, `tab-lab`).
  - Design guardrails: Vanilla CSS with CSS custom properties (`--baize-*`, `--card-*`), 4-color suit classes, responsive `@media` rules, no Tailwind CDN, no forbidden purple-on-dark hex/HSL palettes.
  - JavaScript validity & pure helper unit checks via `node --check src/tenace/web/app.js` (when `node` is present) and end-to-end HTTP smoke flow (`GET /` -> `GET /health` -> `POST /v1/table/state` -> `POST /v1/step` -> `POST /v1/match`).
- [ ] **Step 2: Run `uv run pytest tests/test_web_ui.py -v` to verify RED failures**
- [ ] **Step 3: Implement `index.html`, `styles.css`, `app.js`, and update `README.md`**
- [ ] **Step 4: Run full `uv run pytest -v` to verify GREEN**
- [ ] **Step 5: Commit Task 2** (`feat(web): build interactive Bridge table UI and Duplicate Match Lab`)
