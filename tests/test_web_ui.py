from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from fastapi.testclient import TestClient

from tenace.cli import ModelProfile
from tenace.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "src" / "tenace" / "web"
INDEX_HTML = WEB_DIR / "index.html"
STYLES_CSS = WEB_DIR / "styles.css"
APP_JS = WEB_DIR / "app.js"
README_MD = REPO_ROOT / "README.md"

REQUIRED_ELEMENT_IDS: set[str] = {
    "mode-tabs",
    "tab-play",
    "tab-lab",
    "view-play",
    "view-lab",
    "table-mode-select",
    "board-number-input",
    "seed-input",
    "random-deal-btn",
    "new-deal-btn",
    "undo-btn",
    "step-ai-btn",
    "autoplay-toggle-btn",
    "reveal-hands-toggle",
    "four-color-toggle",
    "seat-north",
    "seat-east",
    "seat-south",
    "seat-west",
    "trick-arena",
    "bidding-box",
    "auction-history-grid",
    "strategy-feed",
    "telemetry-strip",
    "error-banner",
    "profile-mode-select",
    "strategist-model-input",
    "evaluator-backend-select",
    "evaluator-model-input",
    "dds-worlds-input",
    "lab-form",
    "lab-run-btn",
    "lab-double-dummy",
    "lab-results",
}


class _DOMInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.ids: list[str] = []
        self.meta_description: str | None = None
        self.in_title = False
        self.title_parts: list[str] = []
        self.h1_count = 0
        self.stylesheet_hrefs: list[str] = []
        self.script_srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        self.tags.append(tag_lower)
        attr_map = {k.lower(): (v or "") for k, v in attrs}

        if "id" in attr_map and attr_map["id"]:
            self.ids.append(attr_map["id"])
        if tag_lower == "title":
            self.in_title = True
        elif tag_lower == "h1":
            self.h1_count += 1
        elif tag_lower == "meta" and attr_map.get("name", "").lower() == "description":
            self.meta_description = attr_map.get("content", "")
        elif tag_lower == "link" and "stylesheet" in attr_map.get("rel", "").lower():
            self.stylesheet_hrefs.append(attr_map.get("href", ""))
        elif tag_lower == "script" and attr_map.get("src"):
            self.script_srcs.append(attr_map["src"])

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)


def _mock_profile(**overrides: Any) -> ModelProfile:
    return ModelProfile(mode="mock", **overrides)


def test_semantic_html_seo_landmarks_and_unique_interactive_ids() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "cdn.tailwindcss.com" not in html.lower()

    inspector = _DOMInspector()
    inspector.feed(html)

    title_text = "".join(inspector.title_parts).strip()
    assert len(title_text) >= 15
    assert "tenace" in title_text.lower()
    assert inspector.meta_description is not None
    assert len(inspector.meta_description.strip()) >= 50
    assert inspector.h1_count == 1

    # Semantic HTML5 landmarks
    for landmark in ("header", "nav", "main", "section", "aside"):
        assert landmark in inspector.tags, f"Missing semantic <{landmark}> landmark"

    # Google Fonts + static asset wiring
    assert any("Fraunces" in href for href in inspector.stylesheet_hrefs)
    assert any("Plus+Jakarta+Sans" in href for href in inspector.stylesheet_hrefs)
    assert any("JetBrains+Mono" in href for href in inspector.stylesheet_hrefs)
    assert "/static/styles.css" in inspector.stylesheet_hrefs
    assert "/static/app.js" in inspector.script_srcs

    # All IDs must be strictly unique and cover every required interactive element
    duplicates = {
        elem_id for elem_id in inspector.ids if inspector.ids.count(elem_id) > 1
    }
    assert not duplicates, f"Duplicate HTML IDs found: {sorted(duplicates)}"

    missing_ids = REQUIRED_ELEMENT_IDS - set(inspector.ids)
    assert not missing_ids, f"Missing required element IDs: {sorted(missing_ids)}"


def test_design_system_tokens_suit_palettes_and_anti_trope_guardrails() -> None:
    css = STYLES_CSS.read_text(encoding="utf-8")

    # Championship club baize, brass, and warm ivory HSL tokens
    for required_token in (
        "--baize-deep: hsl(158, 46%, 11%)",
        "--baize-mid: hsl(156, 42%, 16%)",
        "--baize-highlight: hsl(154, 38%, 21%)",
        "--brass: hsl(41, 62%, 56%)",
        "--card-ivory: hsl(42, 38%, 97%)",
    ):
        assert required_token in css, f"Missing design token: {required_token}"

    # 4-color suit classes + classic 2-color deck override + interactive legal state
    for selector_fragment in (
        ".suit-spades",
        ".suit-hearts",
        ".suit-diamonds",
        ".suit-clubs",
        "classic-deck",
        ".is-legal",
        "@media",
        "prefers-reduced-motion",
    ):
        assert selector_fragment in css, f"Missing CSS rule fragment: {selector_fragment}"

    # Unused selectors removed & mobile .contract-plaque min-width reset present
    for unused_selector in (".hand-suit-rows", ".suit-row", ".suit-row-pip"):
        assert unused_selector not in css, f"Unused CSS selector still present: {unused_selector}"

    mobile_block_match = re.search(
        r"@media\s*\(\s*max-width:\s*860px\s*\)\s*\{(?P<body>.*)\}\s*@media\s*\(\s*prefers-reduced-motion",
        css,
        flags=re.DOTALL,
    )
    assert mobile_block_match is not None, "Missing @media (max-width: 860px) block"
    assert re.search(
        r"\.contract-plaque\s*\{[^}]*min-width:\s*0\s*;",
        mobile_block_match.group("body"),
        flags=re.DOTALL,
    ), "Missing min-width: 0 on .contract-plaque inside @media (max-width: 860px)"

    # Anti-trope checks: zero purple/violet hues or gradient-text clip tropes
    assert "-webkit-background-clip: text" not in css.lower()
    assert "background-clip: text" not in css.lower()
    for forbidden_word in ("purple", "violet", "indigo", "#7c3aed", "#8b5cf6", "#6366f1"):
        assert forbidden_word not in css.lower()

    hsl_hues = [
        int(m.group(1))
        for m in re.finditer(r"hsla?\(\s*(\d{1,3})\b", css, flags=re.IGNORECASE)
    ]
    assert len(hsl_hues) >= 8
    purple_hues = [hue for hue in hsl_hues if 250 <= hue <= 320]
    assert not purple_hues, f"Forbidden purple/violet HSL hues found: {purple_hues}"


def test_health_exposes_structured_model_profile_for_client_autoconfig() -> None:
    profile = ModelProfile(
        mode="vertex",
        project="sample-cloud-project",
        strategist_location="global",
        strategist_model="gemini-4-argon",
        evaluator_backend="gemini",
        evaluator_location="global",
        evaluator_model="gemini-3-flash-preview",
    )
    client = TestClient(create_app(default_profile=profile, default_dds_worlds=8))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mode"] == "vertex"
    assert body["dds_worlds"] == 8
    assert body["profile"] == profile.model_dump(mode="json")


def test_javascript_syntax_and_pure_helper_unit_behavior() -> None:
    node_bin = shutil.which("node") or "/usr/bin/node"
    assert Path(node_bin).is_file(), "Node.js binary is required for JS syntax verification"

    syntax_proc = subprocess.run(
        [node_bin, "--check", str(APP_JS)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax_proc.returncode == 0, f"node --check failed:\n{syntax_proc.stderr}"

    node_test_script = """
const assert = require("node:assert/strict");
const helpers = require(process.argv[1]);

// 1. Seed clamping & random seed bounds
assert.equal(helpers.clampSeed(42), 42);
assert.equal(helpers.clampSeed(-99), 0);
assert.equal(helpers.clampSeed(99999999999), 2147483647);
assert.equal(helpers.clampSeed("not-a-number"), 42);
for (let i = 0; i < 25; i++) {
  const s = helpers.randomSafeSeed();
  assert.equal(Number.isInteger(s), true);
  assert.equal(s >= 1 && s <= 1000000, true);
}

// 2. Card & call display parsing
const spadeAce = helpers.parseCardDisplay("S:A");
assert.equal(spadeAce.suit, "S");
assert.equal(spadeAce.symbol, "♠");
assert.equal(spadeAce.displayRank, "A");
assert.equal(spadeAce.suitClass, "suit-spades");

const heartTen = helpers.parseCardDisplay("H:T");
assert.equal(heartTen.suit, "H");
assert.equal(heartTen.symbol, "♥");
assert.equal(heartTen.displayRank, "10");
assert.equal(heartTen.suitClass, "suit-hearts");

assert.equal(helpers.formatCallDisplay("Pass").display, "Pass");
assert.equal(helpers.formatCallDisplay("Dbl").display, "X");
assert.equal(helpers.formatCallDisplay("RDbl").display, "XX");
assert.equal(helpers.formatCallDisplay("3NT").display, "3NT");
assert.equal(helpers.formatCallDisplay("4S").symbol, "♠");

// 3. WBF 4-column (WEST | NORTH | EAST | SOUTH) auction alignment padding
const calls = [
  { seat: "SOUTH", label: "1NT", action_id: 59 },
  { seat: "WEST", label: "Pass", action_id: 52 },
  { seat: "NORTH", label: "3NT", action_id: 69 },
  { seat: "EAST", label: "Pass", action_id: 52 },
  { seat: "SOUTH", label: "Pass", action_id: 52 },
  { seat: "WEST", label: "Pass", action_id: 52 },
];
const rowsSouthDealer = helpers.buildAuctionRows("SOUTH", calls);
assert.equal(rowsSouthDealer.length, 3);
assert.deepEqual(
  rowsSouthDealer[0].map((c) => (c ? c.label : null)),
  [null, null, null, "1NT"]
);
assert.deepEqual(
  rowsSouthDealer[1].map((c) => (c ? c.label : null)),
  ["Pass", "3NT", "Pass", "Pass"]
);
assert.deepEqual(
  rowsSouthDealer[2].map((c) => (c ? c.label : null)),
  ["Pass", null, null, null]
);

// 4. Human turn detection (including Declarer South playing Dummy North, and NOT when North plays Dummy South!)
assert.equal(
  helpers.isHumanDecisionTurn("human_vs_tenace", {
    phase: "auction",
    current_player: "SOUTH",
    acting_hand: "SOUTH",
  }),
  true
);
assert.equal(
  helpers.isHumanDecisionTurn("human_vs_baseline", {
    phase: "play",
    current_player: "SOUTH",
    acting_hand: "NORTH",
  }),
  true
);
assert.equal(
  helpers.isHumanDecisionTurn("human_vs_tenace", {
    phase: "play",
    current_player: "NORTH",
    acting_hand: "SOUTH",
  }),
  false
);
assert.equal(
  helpers.isHumanDecisionTurn("spectator_tenace", {
    phase: "play",
    current_player: "SOUTH",
    acting_hand: "SOUTH",
  }),
  false
);

// 5. Bot routing per table mode
assert.equal(helpers.resolveBotForSeat("human_vs_tenace", "NORTH"), "tenace");
assert.equal(helpers.resolveBotForSeat("human_vs_tenace", "EAST"), "tenace");
assert.equal(helpers.resolveBotForSeat("human_vs_baseline", "NORTH"), "tenace");
assert.equal(helpers.resolveBotForSeat("human_vs_baseline", "EAST"), "baseline");
assert.equal(helpers.resolveBotForSeat("human_vs_baseline", "WEST"), "baseline");
assert.equal(helpers.resolveBotForSeat("spectator_tenace", "SOUTH"), "tenace");

// 6. Undo target length: rewinds to previous South decision in human mode, 1 step in spectator mode
const timeline = [
  { actorPlayer: "NORTH" },
  { actorPlayer: "EAST" },
  { actorPlayer: "SOUTH" },
  { actorPlayer: "WEST" },
  { actorPlayer: "NORTH" },
];
assert.equal(helpers.computeUndoTargetLength("spectator_tenace", timeline), 4);
assert.equal(helpers.computeUndoTargetLength("human_vs_tenace", timeline), 2);

// 7. Profile override builder preserves Cloud Run project and omits unchanged default profile
const serverProf = {
  mode: "vertex",
  project: "cloud-run-active-proj",
  strategist_location: "global",
  strategist_model: "gemini-4-argon",
  evaluator_backend: "gemini",
  evaluator_location: "global",
  evaluator_model: "gemini-3-flash-preview",
  evaluator_endpoint_id: null,
  decisions_auth: "vertex",
  decisions_violation_threshold: 0.7,
};
assert.equal(
  helpers.buildProfilePayload(
    {
      mode: "vertex",
      strategist_model: "gemini-4-argon",
      evaluator_backend: "gemini",
      evaluator_model: "gemini-3-flash-preview",
    },
    serverProf
  ),
  null
);
const overridden = helpers.buildProfilePayload(
  {
    mode: "vertex",
    strategist_model: "gemini-2.5-pro",
    evaluator_backend: "diffusion-gemma",
    evaluator_model: "publishers/google/models/diffusiongemma-26b-a4b-it",
  },
  serverProf
);
assert.equal(overridden.project, "cloud-run-active-proj");
assert.equal(overridden.strategist_model, "gemini-2.5-pro");
assert.equal(overridden.evaluator_backend, "diffusion-gemma");
assert.equal(overridden.evaluator_location, "us-central1");

// 8. Whitespace-only model overrides trim BEFORE fallback (never emitting blank model names)
assert.equal(
  helpers.buildProfilePayload(
    {
      mode: "  vertex  ",
      strategist_model: "   ",
      evaluator_backend: "  gemini  ",
      evaluator_model: "   ",
    },
    serverProf,
    false
  ),
  null
);
const blankFallbackExplicit = helpers.buildProfilePayload(
  {
    mode: "   ",
    strategist_model: "   ",
    evaluator_backend: "   ",
    evaluator_model: "   ",
  },
  { ...serverProf, evaluator_backend: "diffusion-gemma", evaluator_model: "custom-diff", evaluator_location: "us-central1", evaluator_endpoint_id: "   " },
  true
);
assert.equal(blankFallbackExplicit.mode, "vertex");
assert.equal(blankFallbackExplicit.strategist_model, "gemini-4-argon");
assert.equal(blankFallbackExplicit.evaluator_backend, "diffusion-gemma");
assert.equal(blankFallbackExplicit.evaluator_model, "custom-diff");
assert.equal(blankFallbackExplicit.evaluator_endpoint_id, null);
console.log(JSON.stringify({ ok: true }));
"""
    unit_proc = subprocess.run(
        [node_bin, "-e", node_test_script, str(APP_JS)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert unit_proc.returncode == 0, f"Node unit tests failed:\n{unit_proc.stderr}"
    assert json.loads(unit_proc.stdout.strip()) == {"ok": True}


def test_browser_controller_concurrency_tab_abort_and_retry_guards() -> None:
    node_bin = shutil.which("node") or "/usr/bin/node"
    dom_harness_script = """
const assert = require("node:assert/strict");
const appPath = process.argv[1];

class FakeClassList {
  constructor() { this.set = new Set(); }
  add(...tokens) { tokens.forEach((t) => this.set.add(t)); }
  remove(...tokens) { tokens.forEach((t) => this.set.delete(t)); }
  toggle(token, force) {
    if (force === undefined) {
      if (this.set.has(token)) { this.set.delete(token); return false; }
      this.set.add(token); return true;
    }
    if (force) { this.set.add(token); } else { this.set.delete(token); }
    return Boolean(force);
  }
  contains(token) { return this.set.has(token); }
}

class FakeElement {
  constructor(id) {
    this.id = id;
    this.value = "";
    this.checked = id === "four-color-toggle" || id === "lab-double-dummy";
    this.disabled = false;
    this.hidden = false;
    this.textContent = "";
    this.innerHTML = "";
    this.attrs = {};
    this.listeners = {};
    this.classList = new FakeClassList();
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(type, fn) {
    if (!this.listeners[type]) this.listeners[type] = [];
    this.listeners[type].push(fn);
  }
  async dispatch(type, ev = {}) {
    const fns = this.listeners[type] || [];
    for (const fn of fns) {
      await fn({ preventDefault() {}, target: ev.target || this, ...ev });
    }
  }
}

const elements = new Map();
function getEl(id) {
  if (!elements.has(id)) {
    const el = new FakeElement(id);
    if (id === "table-mode-select") el.value = "human_vs_tenace";
    if (id === "board-number-input") el.value = "3";
    if (id === "seed-input") el.value = "42";
    if (id === "dds-worlds-input") el.value = "4";
    if (id === "profile-mode-select") el.value = "mock";
    if (id === "strategist-model-input") el.value = "gemini-4-argon";
    if (id === "evaluator-backend-select") el.value = "gemini";
    if (id === "evaluator-model-input") el.value = "gemini-custom-eval";
    if (id === "lab-boards-input") el.value = "2";
    if (id === "lab-seed-input") el.value = "42";
    if (id === "lab-dds-worlds-input") el.value = "4";
    if (id === "lab-opponent-select") el.value = "tenace";
    elements.set(id, el);
  }
  return elements.get(id);
}

const fetchQueue = [];
const fetchCalls = [];
function enqueueFetch(matcher, handler) {
  fetchQueue.push({ matcher, handler });
}

global.document = {
  body: new FakeElement("body"),
  getElementById: (id) => getEl(id),
};
global.window = {
  performance: { now: () => 1000 },
  setTimeout: () => 1,
  clearTimeout: () => {},
  fetch: (url, opts = {}) => {
    fetchCalls.push({ url, opts });
    const idx = fetchQueue.findIndex((item) => item.matcher === url);
    assert.notEqual(idx, -1, `Unexpected fetch to ${url}`);
    const { handler } = fetchQueue.splice(idx, 1)[0];
    return handler(url, opts);
  },
};

function makeJsonResponse(status, body) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    text: () => Promise.resolve(JSON.stringify(body)),
  });
}

const southTurnState = {
  tournament_seed: 42,
  board_number: 1,
  dealer: "SOUTH",
  ns_vulnerable: false,
  ew_vulnerable: false,
  phase: "auction",
  viewer_seat: "SOUTH",
  reveal_all: false,
  current_player: "SOUTH",
  acting_hand: "SOUTH",
  hands: { NORTH: null, EAST: null, SOUTH: ["S:A"], WEST: null },
  hand_hcp: { NORTH: null, EAST: null, SOUTH: 4, WEST: null },
  remaining_counts: { NORTH: 13, EAST: 13, SOUTH: 13, WEST: 13 },
  initial_hands: null,
  auction_calls: [],
  contract: null,
  ns_tricks: 0,
  ew_tricks: 0,
  current_trick: [],
  completed_tricks: [],
  legal_actions: [{ action_id: 52, label: "Pass" }],
  terminal_result: null,
};

const westAiTurnState = {
  ...southTurnState,
  current_player: "WEST",
  acting_hand: "WEST",
  auction_calls: [{ seat: "SOUTH", label: "Pass", action_id: 52 }],
};

const northAiTurnState = {
  ...southTurnState,
  current_player: "NORTH",
  acting_hand: "NORTH",
  auction_calls: [
    { seat: "SOUTH", label: "Pass", action_id: 52 },
    { seat: "WEST", label: "Pass", action_id: 52 },
  ],
};

(async () => {
  // Initial boot: /health + /v1/table/state
  enqueueFetch("/health", () =>
    makeJsonResponse(200, {
      status: "ok",
      service: "tenace",
      version: "0.1.0",
      default_profile: "mock | L1:gemini-4-argon | L2:gemini/gemini-custom-eval",
      mode: "mock",
      dds_worlds: 4,
      profile: {
        mode: "mock",
        project: "proj-test",
        strategist_location: "global",
        strategist_model: "gemini-4-argon",
        evaluator_backend: "gemini",
        evaluator_location: "global",
        evaluator_model: "gemini-custom-eval",
        evaluator_endpoint_id: null,
        decisions_auth: "vertex",
        decisions_violation_threshold: 0.7,
      },
    })
  );
  enqueueFetch("/v1/table/state", () => makeJsonResponse(200, southTurnState));

  require(appPath);
  await new Promise((r) => setImmediate(r));
  await new Promise((r) => setImmediate(r));

  // Disable Auto-Play so we control AI steps deterministically
  await getEl("autoplay-toggle-btn").dispatch("click");

  // 1. #lab-team-b-backend change passes state.serverProfile (resolving gemini-custom-eval!)
  getEl("lab-team-b-backend").value = "decisions";
  await getEl("lab-team-b-backend").dispatch("change");
  assert.equal(getEl("lab-team-b-evaluator").value, "gemini-decision-1");
  getEl("lab-team-b-backend").value = "gemini";
  await getEl("lab-team-b-backend").dispatch("change");
  assert.equal(getEl("lab-team-b-evaluator").value, "gemini-custom-eval");

  // 2. Double-click Pass race guard: while /v1/table/state is in-flight, 2nd human bid MUST be ignored & MUST NOT abort 1st fetch!
  let resolveHumanStateSync;
  let firstHumanSignal;
  enqueueFetch("/v1/table/state", (_url, opts) => {
    firstHumanSignal = opts.signal;
    return new Promise((resolve) => {
      resolveHumanStateSync = () => resolve(makeJsonResponse(200, westAiTurnState));
    });
  });

  const fakePassBtn = {
    disabled: false,
    getAttribute: (name) => (name === "data-bid-action-id" ? "52" : null),
    closest: () => fakePassBtn,
  };

  // First click starts refreshTableState (in-flight)
  const click1 = getEl("bidding-box").dispatch("click", { target: fakePassBtn });
  await new Promise((r) => setImmediate(r));
  assert.equal(firstHumanSignal.aborted, false);
  const callsAfterClick1 = fetchCalls.length;

  // Rapid second click before DOM/fetch settles must bail out cleanly without aborting click1 or pushing duplicate action!
  await getEl("bidding-box").dispatch("click", { target: fakePassBtn });
  assert.equal(firstHumanSignal.aborted, false);
  assert.equal(fetchCalls.length, callsAfterClick1);

  resolveHumanStateSync();
  await click1;
  await new Promise((r) => setImmediate(r));
  const syncBodyAfterPass = JSON.parse(fetchCalls[callsAfterClick1 - 1].opts.body);
  assert.deepEqual(syncBodyAfterPass.action_history, [52]);

  // 3. executeAiStep keeps isBusy=true across follow-up refreshTableState, and if refreshTableState fails,
  //    Retry Step retries refreshTableState WITHOUT wiping the in-progress board!
  enqueueFetch("/v1/step", () =>
    makeJsonResponse(200, {
      action_id: 52,
      action_label: "Pass",
      seat: "WEST",
      phase: "auction",
      strategy: null,
      telemetry: { strategist_calls: 0, evaluator_calls: 0, forced_moves: 0 },
    })
  );
  let rejectPostStepSync;
  enqueueFetch("/v1/table/state", () =>
    new Promise((_resolve, reject) => {
      rejectPostStepSync = () => reject(new Error("Transient 502 during state sync"));
    })
  );

  const stepPromise = getEl("step-ai-btn").dispatch("click");
  await new Promise((r) => setImmediate(r));
  // While follow-up /v1/table/state is still in-flight after /v1/step, Step AI & Undo MUST still be disabled (isBusy=true)!
  assert.equal(getEl("step-ai-btn").disabled, true);
  assert.equal(getEl("undo-btn").disabled, true);

  rejectPostStepSync();
  await stepPromise;
  await new Promise((r) => setImmediate(r));
  assert.equal(getEl("error-banner").hidden, false);

  // Clicking Retry Step must call /v1/table/state with BOTH actions [52, 52] preserved (NOT startNewDeal [] and NOT /v1/step)!
  enqueueFetch("/v1/table/state", () => makeJsonResponse(200, northAiTurnState));
  await getEl("retry-step-btn").dispatch("click");
  await new Promise((r) => setImmediate(r));
  const retryFetch = fetchCalls[fetchCalls.length - 1];
  assert.equal(retryFetch.url, "/v1/table/state");
  assert.deepEqual(JSON.parse(retryFetch.opts.body).action_history, [52, 52]);

  // 4. switchTab("lab") aborts an in-flight Tab 1 /v1/step immediately
  let inFlightStepSignal;
  enqueueFetch("/v1/step", (_url, opts) => {
    inFlightStepSignal = opts.signal;
    return new Promise(() => {}); // hangs until aborted
  });
  getEl("step-ai-btn").dispatch("click");
  await new Promise((r) => setImmediate(r));
  assert.equal(inFlightStepSignal.aborted, false);

  await getEl("tab-lab").dispatch("click");
  assert.equal(inFlightStepSignal.aborted, true);

  // 5. handleRunLabMatch guards against double-submit via isLabBusy
  let resolveMatch;
  enqueueFetch("/v1/match", () =>
    new Promise((resolve) => {
      resolveMatch = () =>
        resolve(
          makeJsonResponse(200, {
            team_a_name: "Tenace-A",
            team_b_name: "Tenace-B",
            team_a_profile: "mock",
            team_b_profile: "mock",
            summary: {
              tournament_seed: 42,
              boards: [],
              total_imps_team_a: 0,
              total_imps_team_b: 0,
              telemetry: {},
            },
            report: "OK",
          })
        );
    })
  );

  const countBeforeMatch = fetchCalls.length;
  const run1 = getEl("lab-form").dispatch("submit");
  await new Promise((r) => setImmediate(r));
  // Second submit while first is in-flight must be ignored!
  await getEl("lab-form").dispatch("submit");
  assert.equal(fetchCalls.length, countBeforeMatch + 1);

  resolveMatch();
  await run1;
  console.log(JSON.stringify({ controllerOk: true }));
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    proc = subprocess.run(
        [node_bin, "-e", dom_harness_script, str(APP_JS)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Browser controller concurrency tests failed:\n{proc.stderr}"
    assert json.loads(proc.stdout.strip()) == {"controllerOk": True}


def test_end_to_end_web_client_api_flow_and_readme_docs() -> None:
    app = create_app(default_profile=_mock_profile(project="demo-proj"), default_dds_worlds=2)
    client = TestClient(app)

    # 1. Browser boot: GET / + GET /health
    assert client.get("/").status_code == 200
    health = client.get("/health").json()
    assert health["profile"]["project"] == "demo-proj"

    # 2. Load initial Deal state on Board 3 (South dealer)
    state_resp = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 3,
            "action_history": [],
            "viewer_seat": "SOUTH",
            "reveal_all": False,
        },
    )
    assert state_resp.status_code == 200
    table = state_resp.json()
    assert table["current_player"] == "SOUTH"
    assert any(a["label"] == "1NT" for a in table["legal_actions"])

    # 3. Human South bids 1NT, Fast Baseline West passes, Tenace North responds
    history: list[str | int] = ["1NT"]
    west_step = client.post(
        "/v1/step",
        json={
            "seat": "WEST",
            "board_number": 3,
            "tournament_seed": 42,
            "action_history": history,
            "bot": "baseline",
        },
    )
    assert west_step.status_code == 200
    assert west_step.json()["action_label"] == "Pass"
    history.append(west_step.json()["action_id"])

    north_step = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 3,
            "tournament_seed": 42,
            "action_history": history,
            "bot": "tenace",
            "dds_worlds": 1,
        },
    )
    assert north_step.status_code == 200
    assert north_step.json()["strategy"] is not None

    # 4. Duplicate Match Lab smoke call
    match_resp = client.post(
        "/v1/match",
        json={
            "boards": 1,
            "seed": 42,
            "dds_worlds": 1,
            "double_dummy": True,
            "opponent": "baseline",
        },
    )
    assert match_resp.status_code == 200
    assert len(match_resp.json()["summary"]["boards"]) == 1

    # 5. README documents Interactive Web Client launch locally and via Cloud Run proxy
    readme_text = README_MD.read_text(encoding="utf-8")
    assert "Interactive Web Client" in readme_text
    assert "http://127.0.0.1:8080/" in readme_text
    assert "gcloud run services proxy" in readme_text

    # 6. Public repository hygiene check across web assets, server, README, and tests
    dynamic_literals = {
        candidate
        for candidate in (str(Path.home()), str(Path.home().parent), Path.home().name)
        if len(candidate) >= 4 and candidate not in {"/home", "/Users", "root", "runner", "ubuntu", "user"}
    }
    forbidden_patterns = (
        re.compile(r"\b[a-z][a-z0-9-]{3,23}-\d{6}\b"),
        re.compile(r"\b[a-z0-9.-]+\.goog\b", re.IGNORECASE),
        re.compile(r"/" + r"google/(?:data|src)/"),
    )
    for path in (
        INDEX_HTML,
        STYLES_CSS,
        APP_JS,
        README_MD,
        REPO_ROOT / "src" / "tenace" / "server.py",
        REPO_ROOT / "tests" / "test_server.py",
        Path(__file__),
    ):
        content = path.read_text(encoding="utf-8")
        for lit in dynamic_literals:
            assert lit not in content, f"Forbidden local identifier found in {path.name}"
        for pat in forbidden_patterns:
            assert pat.search(content) is None, f"Forbidden internal pattern found in {path.name}"
