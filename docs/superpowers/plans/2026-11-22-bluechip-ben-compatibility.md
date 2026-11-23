# BlueChip v18 Protocol Compatibility for Real Bridge Engines (BEN / WBridge5)

**Goal:** Align Tenace's BlueChip v18 Table Manager wire adapter (`src/tenace/harness/bluechip.py`) and CLI launcher (`src/tenace/cli.py`) with the official WBF BlueChip v18 specification verified against `lorserker/ben` (`table_manager_client.py` v0.9.0.4), following strict TDD (`RED -> GREEN`).

## Root-Cause Protocol Mismatches Found

1. **Suit ordering in `format_bluechip_hand` (`src/tenace/harness/bluechip.py`):**
   - Previously emitted `C ... D ... H ... S ...` because `SUIT_CHARS` is ordered ascending `("C", "D", "H", "S")`.
   - The BlueChip v18 wire protocol and BEN `TMClient.parse_hand()` expect **descending rank-order suits**: `S ... H ... D ... C ...` (e.g. `S J T 8 6 4. H Q 9 7 5 3. D -. C A K 2.`). Reversing suit order caused real peers to silently swap spades/clubs and hearts/diamonds.

2. **`Teams` line colons (`BlueChipSeatBot._ensure_connected`):**
   - Previously emitted `Teams: N/S "<ns>" E/W "<ew>"`.
   - Official BlueChip Table Manager and BEN's parser regex (`r'N/S : "([^"]+)" E/W : "([^"]+)"(?:.*?Playing (\w+))?'`) require spaces/colons after `N/S` and `E/W`: `Teams : N/S : "<ns>" E/W : "<ew>"`.

3. **Title-Case seat names on the wire:**
   - Previously emitted uppercase seat names (`NORTH`, `EAST`, `SOUTH`, `WEST`).
   - Real BlueChip v18 peers (`table_manager_client.py`) perform case-sensitive equality assertions against `['North', 'East', 'South', 'West']` in `receive_bid_for` and `receive_card_play_for`.
   - Update outbound wire messages (`seated`, `format_bluechip_deal_header`, `'<Seat>\'s cards'`, bid/play notifications, `'<Seat> to lead'`) to Title Case (`North`, `East`, `South`, `West`). Inbound parsing stays case-insensitive to remain compatible with both uppercase and Title-Case engines.

4. **Trailing metadata / period tolerance in `_CONNECT_RE` and `_PLAYER_ACTION_RE`:**
   - Allow optional trailing metadata after `protocol version <n>` (e.g. BEN `--sendinfo True`).
   - Allow an optional trailing period / info suffix on player action lines (e.g. `East plays 2S. Forced` from BEN `--sendcardinfo True`) without accepting malformed card/bid tokens (such as `2Sx` or `8NT`).

5. **Optional `{seat}` placeholder in `SubprocessTcpBlueChipController` & `tenace-match` CLI:**
   - Engines such as BEN require `--seat North|East|South|West` at process launch.
   - Add optional `seat: Seat | None = None` to `SubprocessTcpBlueChipController.__init__` that substitutes `{seat}` with Title-Case compass seat (`North`, `East`, `South`, `West`), raising `ValueError` if `{seat}` appears in `cmd_template` without a `seat`.
   - Pass `seat=seat` from `run_cli` when launching `--opponent bluechip` subprocesses and document `{seat}` in `--bluechip-cmd` help text.

## TDD Task Breakdown

### Task 1: RED — Add & update protocol compliance tests
- Files: `tests/test_bluechip_adapter.py`, `tests/test_cli.py`
- Verify `format_bluechip_hand` outputs `S ... H ... D ... C ...` including voids (`-`).
- Verify `format_bluechip_deal_header` outputs Title-Case dealer names (`North`, `East`, `South`, `West`).
- Verify `BlueChipSeatBot` emits `Teams : N/S : "<ns>" E/W : "<ew>"` and Title-Case seat names on `seated`, cards, bid/play relays, and `to lead`.
- Verify `_CONNECT_RE` tolerates optional trailing config string after protocol version, and `parse_bluechip_player_action` accepts trailing period/card-info (`East plays 2S. Forced`, `North passes.`) while still rejecting invalid actions (`North plays 2Sx`, `North bids 8NT`).
- Verify `SubprocessTcpBlueChipController` expands `{seat}` -> `North`/`East`/`South`/`West` and rejects `{seat}` without `seat` argument.
- Verify `LoopbackBlueChipPeer` and `_BLUECHIP_SUBPROCESS_PEER` validate exact Title-Case wire names, `S H D C` suit order, and `Teams : N/S : ... E/W : ...` format, and run `tenace-match --opponent bluechip` with `{seat}`.
- Run `pytest tests/test_bluechip_adapter.py tests/test_cli.py` and confirm expected `RED` failures.

### Task 2: GREEN — Implement minimal BlueChip v18 fixes
- Files: `src/tenace/harness/bluechip.py`, `src/tenace/cli.py`
- Update `format_bluechip_hand`, `format_bluechip_deal_header`, `_wire_seat_name` helper, `Teams` line format, `_CONNECT_RE`, `_PLAYER_ACTION_RE`, `SubprocessTcpBlueChipController.__init__`, and `run_cli` factory wiring.
- Run targeted suite `pytest tests/test_bluechip_adapter.py tests/test_cli.py` -> all `GREEN`.

### Task 3: Full Verification & Live Integration Check
- Run full `uv run pytest -q` regression suite (133+ tests).
- Smoke-test against real `lorserker/ben` (`table_manager_client.py`) over real loopback TCP with `--mode mock` (offline protocol validation across bidding, passout, dummy reveal, card play, and multi-board transitions).
- Run a live duplicate IMP match between Tenace (`Vertex AI IAM`) and BEN.
- Perform public-repo hygiene check (`git diff`) before committing/pushing.
