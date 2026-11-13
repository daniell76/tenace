"use strict";

(() => {
  const MAX_INT32 = 2147483647;
  const AUCTION_SEAT_COLUMNS = ["WEST", "NORTH", "EAST", "SOUTH"];
  const SEAT_INDEX_NAMES = {
    0: "NORTH",
    1: "EAST",
    2: "SOUTH",
    3: "WEST",
    NORTH: "NORTH",
    EAST: "EAST",
    SOUTH: "SOUTH",
    WEST: "WEST",
  };
  const SUIT_META = {
    S: { symbol: "♠", suitClass: "suit-spades", name: "Spades" },
    H: { symbol: "♥", suitClass: "suit-hearts", name: "Hearts" },
    D: { symbol: "♦", suitClass: "suit-diamonds", name: "Diamonds" },
    C: { symbol: "♣", suitClass: "suit-clubs", name: "Clubs" },
  };
  const SUIT_ENUM_SYMBOLS = {
    0: "♣",
    1: "♦",
    2: "♥",
    3: "♠",
    CLUBS: "♣",
    DIAMONDS: "♦",
    HEARTS: "♥",
    SPADES: "♠",
    C: "♣",
    D: "♦",
    H: "♥",
    S: "♠",
  };

  /* ========================================================================
     Pure Helpers (Exported for Node Unit Tests & Used by Browser UI)
     ======================================================================== */

  function escapeHtml(raw) {
    return String(raw ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function clampSeed(value, fallback = 42) {
    const parsed = typeof value === "number" ? value : Number.parseInt(String(value ?? ""), 10);
    if (!Number.isFinite(parsed) || Number.isNaN(parsed)) {
      return fallback;
    }
    const truncated = Math.trunc(parsed);
    if (truncated < 0) {
      return 0;
    }
    if (truncated > MAX_INT32) {
      return MAX_INT32;
    }
    return truncated;
  }

  function clampInt(value, min, max, fallback) {
    const parsed = typeof value === "number" ? value : Number.parseInt(String(value ?? ""), 10);
    if (!Number.isFinite(parsed) || Number.isNaN(parsed)) {
      return fallback;
    }
    return Math.min(max, Math.max(min, Math.trunc(parsed)));
  }

  function randomSafeSeed() {
    return Math.floor(Math.random() * 1000000) + 1;
  }

  function parseCardDisplay(cardLabel) {
    const raw = String(cardLabel || "").trim().toUpperCase();
    const parts = raw.split(":");
    const suit = parts[0] || "S";
    const rank = parts[1] || "?";
    const meta = SUIT_META[suit] || SUIT_META.S;
    const displayRank = rank === "T" ? "10" : rank;
    return {
      raw,
      suit,
      rank,
      displayRank,
      symbol: meta.symbol,
      suitClass: meta.suitClass,
      ariaLabel: `${displayRank} of ${meta.name}`,
    };
  }

  function formatCallDisplay(callLabel) {
    const raw = String(callLabel || "").trim();
    if (raw === "Pass") {
      return { raw, display: "Pass", level: "", symbol: "", suitClass: "" };
    }
    if (raw === "Dbl") {
      return { raw, display: "X", level: "", symbol: "X", suitClass: "" };
    }
    if (raw === "RDbl") {
      return { raw, display: "XX", level: "", symbol: "XX", suitClass: "" };
    }
    const level = raw.slice(0, 1);
    const denom = raw.slice(1).toUpperCase();
    if (denom === "NT") {
      return { raw, display: `${level}NT`, level, symbol: "NT", suitClass: "" };
    }
    const meta = SUIT_META[denom];
    if (meta) {
      return {
        raw,
        display: `${level}${meta.symbol}`,
        level,
        symbol: meta.symbol,
        suitClass: meta.suitClass,
      };
    }
    return { raw, display: raw, level: "", symbol: "", suitClass: "" };
  }

  function buildAuctionRows(dealer, auctionCalls) {
    const dealerUpper = String(dealer || "NORTH").toUpperCase();
    const padCount = Math.max(0, AUCTION_SEAT_COLUMNS.indexOf(dealerUpper));
    const cells = [];
    for (let i = 0; i < padCount; i += 1) {
      cells.push(null);
    }
    for (const call of auctionCalls || []) {
      cells.push(call);
    }
    if (cells.length === 0) {
      return [];
    }
    const rows = [];
    for (let idx = 0; idx < cells.length; idx += 4) {
      const row = [
        cells[idx] ?? null,
        cells[idx + 1] ?? null,
        cells[idx + 2] ?? null,
        cells[idx + 3] ?? null,
      ];
      rows.push(row);
    }
    return rows;
  }

  function isHumanDecisionTurn(tableMode, tableState) {
    if (!tableState || tableState.phase === "terminal") {
      return false;
    }
    if (tableMode === "spectator_tenace") {
      return false;
    }
    return tableState.current_player === "SOUTH";
  }

  function resolveBotForSeat(tableMode, seat) {
    if (tableMode === "human_vs_baseline" && (seat === "EAST" || seat === "WEST")) {
      return "baseline";
    }
    return "tenace";
  }

  function computeUndoTargetLength(tableMode, timelineEntries) {
    const entries = Array.isArray(timelineEntries) ? timelineEntries : [];
    if (entries.length === 0) {
      return 0;
    }
    if (tableMode === "spectator_tenace") {
      return entries.length - 1;
    }
    for (let idx = entries.length - 1; idx >= 0; idx -= 1) {
      if (entries[idx] && entries[idx].actorPlayer === "SOUTH") {
        return idx;
      }
    }
    return 0;
  }

  function defaultEvaluatorModelForBackend(backend, serverProfile) {
    if (serverProfile && serverProfile.evaluator_backend === backend && serverProfile.evaluator_model) {
      return serverProfile.evaluator_model;
    }
    if (backend === "decisions") {
      return "gemini-decision-1";
    }
    if (backend === "diffusion-gemma") {
      return "publishers/google/models/diffusiongemma-26b-a4b-it";
    }
    return "gemini-3-flash-preview";
  }

  function resolveEvaluatorLocation(backend, serverProfile) {
    if (serverProfile && serverProfile.evaluator_backend === backend && serverProfile.evaluator_location) {
      return serverProfile.evaluator_location;
    }
    return backend === "diffusion-gemma" ? "us-central1" : "global";
  }

  function buildProfilePayload(uiConfig, serverProfile, forceExplicit = false) {
    const base = serverProfile || {
      mode: "vertex",
      project: null,
      strategist_location: "global",
      strategist_model: "gemini-4-argon",
      evaluator_backend: "decisions",
      evaluator_location: "global",
      evaluator_model: "gemini-decision-1",
      evaluator_endpoint_id: null,
      decisions_auth: "vertex",
      decisions_violation_threshold: 0.7,
    };

    const mode = (uiConfig.mode || base.mode || "vertex").trim();
    const strategistModel = (uiConfig.strategist_model || base.strategist_model || "gemini-4-argon").trim();
    const evaluatorBackend = (uiConfig.evaluator_backend || base.evaluator_backend || "decisions").trim();
    const evaluatorModel = (
      uiConfig.evaluator_model ||
      defaultEvaluatorModelForBackend(evaluatorBackend, base)
    ).trim();

    const unchanged =
      !forceExplicit &&
      serverProfile &&
      mode === serverProfile.mode &&
      strategistModel === serverProfile.strategist_model &&
      evaluatorBackend === serverProfile.evaluator_backend &&
      evaluatorModel === serverProfile.evaluator_model;

    if (unchanged) {
      return null;
    }

    return {
      mode,
      project: base.project ?? null,
      strategist_location: base.strategist_location || "global",
      strategist_model: strategistModel,
      evaluator_backend: evaluatorBackend,
      evaluator_location: resolveEvaluatorLocation(evaluatorBackend, base),
      evaluator_model: evaluatorModel,
      evaluator_endpoint_id:
        evaluatorBackend === "diffusion-gemma" ? (base.evaluator_endpoint_id ?? null) : null,
      decisions_auth:
        evaluatorBackend === "decisions" ? (base.decisions_auth || "vertex") : "vertex",
      decisions_violation_threshold:
        typeof base.decisions_violation_threshold === "number"
          ? base.decisions_violation_threshold
          : 0.7,
    };
  }

  const exportedHelpers = {
    escapeHtml,
    clampSeed,
    clampInt,
    randomSafeSeed,
    parseCardDisplay,
    formatCallDisplay,
    buildAuctionRows,
    isHumanDecisionTurn,
    resolveBotForSeat,
    computeUndoTargetLength,
    defaultEvaluatorModelForBackend,
    resolveEvaluatorLocation,
    buildProfilePayload,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = exportedHelpers;
  }

  if (typeof document === "undefined" || typeof window === "undefined") {
    return;
  }

  /* ========================================================================
     Browser Interactive Application State & Controllers
     ======================================================================== */

  const state = {
    activeTab: "play",
    serverHealth: null,
    serverProfile: null,
    serverDdsWorlds: 8,

    tableMode: "human_vs_tenace",
    boardNumber: 1,
    tournamentSeed: 42,
    autoPlayAi: true,
    revealHands: false,
    fourColorDeck: true,
    hideEwThoughts: false,
    inspectingLastTrick: false,

    // Timeline of applied actions: { action, actorPlayer, priorStrategiesBefore, telemetryDelta, feedEntryId }
    timeline: [],
    priorStrategies: { NORTH: null, EAST: null, SOUTH: null, WEST: null },
    telemetryTotals: { strategist_calls: 0, evaluator_calls: 0, forced_moves: 0 },
    lastStepLatencyMs: null,
    strategyFeed: [],
    nextFeedId: 1,

    tableState: null,
    prevCompletedTrickCount: 0,
    isBusy: false,
    sessionEpoch: 0,
    activeAbortController: null,
    autoPlayTimerId: null,
  };

  const dom = {
    serverDot: document.getElementById("server-status-dot"),
    serverText: document.getElementById("server-status-text"),
    tabPlay: document.getElementById("tab-play"),
    tabLab: document.getElementById("tab-lab"),
    viewPlay: document.getElementById("view-play"),
    viewLab: document.getElementById("view-lab"),

    tableModeSelect: document.getElementById("table-mode-select"),
    boardNumberInput: document.getElementById("board-number-input"),
    seedInput: document.getElementById("seed-input"),
    newDealBtn: document.getElementById("new-deal-btn"),
    randomDealBtn: document.getElementById("random-deal-btn"),
    undoBtn: document.getElementById("undo-btn"),
    stepAiBtn: document.getElementById("step-ai-btn"),
    autoPlayBtn: document.getElementById("autoplay-toggle-btn"),
    revealHandsToggle: document.getElementById("reveal-hands-toggle"),
    fourColorToggle: document.getElementById("four-color-toggle"),

    errorBanner: document.getElementById("error-banner"),
    errorMessageText: document.getElementById("error-message-text"),
    retryStepBtn: document.getElementById("retry-step-btn"),
    dismissErrorBtn: document.getElementById("dismiss-error-btn"),

    turnStatusGroup: document.getElementById("turn-status-group"),
    turnSpinner: document.getElementById("turn-spinner"),
    turnStatusText: document.getElementById("turn-status-text"),
    turnPhasePill: document.getElementById("turn-phase-pill"),
    turnLatencyPill: document.getElementById("turn-latency-pill"),

    plaqueBoardTitle: document.getElementById("plaque-board-title"),
    plaqueDealerVul: document.getElementById("plaque-dealer-vul"),
    contractPlaqueValue: document.getElementById("contract-plaque-value"),
    contractPlaqueTricks: document.getElementById("contract-plaque-tricks"),

    seatZones: {
      NORTH: document.getElementById("seat-north"),
      EAST: document.getElementById("seat-east"),
      SOUTH: document.getElementById("seat-south"),
      WEST: document.getElementById("seat-west"),
    },
    trickArena: document.getElementById("trick-arena"),

    auctionTrayWrapper: document.getElementById("auction-tray-wrapper"),
    auctionDealerNote: document.getElementById("auction-dealer-note"),
    auctionHistoryGrid: document.getElementById("auction-history-grid"),
    biddingBox: document.getElementById("bidding-box"),
    playTricksPanel: document.getElementById("play-tricks-panel"),
    showLastTrickBtn: document.getElementById("show-last-trick-btn"),
    trickHistoryList: document.getElementById("trick-history-list"),

    statStrategistCalls: document.getElementById("stat-strategist-calls"),
    statEvaluatorCalls: document.getElementById("stat-evaluator-calls"),
    statForcedMoves: document.getElementById("stat-forced-moves"),
    statLastMs: document.getElementById("stat-last-ms"),

    profileModeSelect: document.getElementById("profile-mode-select"),
    ddsWorldsInput: document.getElementById("dds-worlds-input"),
    strategistModelInput: document.getElementById("strategist-model-input"),
    evaluatorBackendSelect: document.getElementById("evaluator-backend-select"),
    evaluatorModelInput: document.getElementById("evaluator-model-input"),
    resetProfileBtn: document.getElementById("reset-profile-btn"),

    hideEwThoughtsToggle: document.getElementById("hide-ew-thoughts-toggle"),
    strategyFeed: document.getElementById("strategy-feed"),

    labForm: document.getElementById("lab-form"),
    labBoardsInput: document.getElementById("lab-boards-input"),
    labSeedInput: document.getElementById("lab-seed-input"),
    labDdsWorldsInput: document.getElementById("lab-dds-worlds-input"),
    labOpponentSelect: document.getElementById("lab-opponent-select"),
    labDoubleDummy: document.getElementById("lab-double-dummy"),
    labTeamAMode: document.getElementById("lab-team-a-mode"),
    labTeamABackend: document.getElementById("lab-team-a-backend"),
    labTeamAStrategist: document.getElementById("lab-team-a-strategist"),
    labTeamAEvaluator: document.getElementById("lab-team-a-evaluator"),
    labTeamBFieldset: document.getElementById("lab-team-b-fieldset"),
    labTeamBMode: document.getElementById("lab-team-b-mode"),
    labTeamBBackend: document.getElementById("lab-team-b-backend"),
    labTeamBStrategist: document.getElementById("lab-team-b-strategist"),
    labTeamBEvaluator: document.getElementById("lab-team-b-evaluator"),
    labRunBtn: document.getElementById("lab-run-btn"),
    labResults: document.getElementById("lab-results"),
  };

  function cancelPendingAsyncWork() {
    state.sessionEpoch += 1;
    if (state.autoPlayTimerId !== null) {
      window.clearTimeout(state.autoPlayTimerId);
      state.autoPlayTimerId = null;
    }
    if (state.activeAbortController) {
      state.activeAbortController.abort();
      state.activeAbortController = null;
    }
    state.isBusy = false;
  }

  function showError(message) {
    dom.errorMessageText.textContent = message;
    dom.errorBanner.hidden = false;
  }

  function clearError() {
    dom.errorBanner.hidden = true;
    dom.errorMessageText.textContent = "";
  }

  function readTableUiProfile() {
    return {
      mode: dom.profileModeSelect.value,
      strategist_model: dom.strategistModelInput.value,
      evaluator_backend: dom.evaluatorBackendSelect.value,
      evaluator_model: dom.evaluatorModelInput.value,
    };
  }

  function applyServerDefaultsToForms() {
    const prof = state.serverProfile;
    if (!prof) {
      return;
    }
    dom.profileModeSelect.value = prof.mode || "vertex";
    dom.strategistModelInput.value = prof.strategist_model || "gemini-4-argon";
    dom.evaluatorBackendSelect.value = prof.evaluator_backend || "decisions";
    dom.evaluatorModelInput.value = prof.evaluator_model || "gemini-decision-1";
    dom.ddsWorldsInput.value = String(clampInt(state.serverDdsWorlds, 1, 64, 8));

    dom.labTeamAMode.value = prof.mode || "vertex";
    dom.labTeamABackend.value = prof.evaluator_backend || "decisions";
    dom.labTeamAStrategist.value = prof.strategist_model || "gemini-4-argon";
    dom.labTeamAEvaluator.value = prof.evaluator_model || "gemini-decision-1";

    const teamBBackend = prof.evaluator_backend || "gemini";
    dom.labTeamBMode.value = prof.mode || "vertex";
    dom.labTeamBBackend.value = teamBBackend;
    dom.labTeamBStrategist.value =
      prof.strategist_model === "gemini-2.5-pro" ? "gemini-4-argon" : "gemini-2.5-pro";
    dom.labTeamBEvaluator.value =
      teamBBackend === "gemini" && prof.evaluator_model !== "gemini-2.5-flash"
        ? "gemini-2.5-flash"
        : defaultEvaluatorModelForBackend(teamBBackend, prof);
  }

  async function fetchJson(url, options = {}) {
    const resp = await window.fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const text = await resp.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (_err) {
        payload = null;
      }
    }
    if (!resp.ok) {
      const detail =
        (payload && typeof payload.detail === "string" && payload.detail) ||
        (payload && Array.isArray(payload.detail) && payload.detail.map((d) => d.msg).join("; ")) ||
        `HTTP ${resp.status} ${resp.statusText}`;
      throw new Error(detail);
    }
    return payload;
  }

  /* ========================================================================
     Rendering Helpers
     ======================================================================== */

  function formatContractHtml(contractObj) {
    if (!contractObj) {
      return "Auction in progress";
    }
    if (!contractObj.level || !contractObj.trump) {
      return escapeHtml(contractObj.formatted || "Passed Out");
    }
    const trump = contractObj.trump;
    const meta = SUIT_META[trump];
    const strainHtml = meta
      ? `<span class="${meta.suitClass}">${meta.symbol}</span>`
      : escapeHtml(trump);
    const dbl = escapeHtml(contractObj.doubled || "");
    const decl = escapeHtml(contractObj.declarer || "");
    return `${contractObj.level}${strainHtml}${dbl} by ${decl}`;
  }

  function seatRoleDescription(seat, ts) {
    if (ts && ts.contract && ts.contract.declarer) {
      if (ts.contract.declarer === seat) {
        return "Declarer";
      }
      if (ts.contract.dummy === seat) {
        return "Dummy";
      }
    }
    if (state.tableMode === "spectator_tenace") {
      return "Tenace AI";
    }
    if (seat === "SOUTH") {
      return "You · Human";
    }
    if (seat === "NORTH") {
      return "Partner · Tenace";
    }
    return state.tableMode === "human_vs_baseline" ? "Defender · Baseline" : "Defender · Tenace";
  }

  function renderCardButtonHtml(cardLabel, { actionId, isLegal, isDimmed, isCompact }) {
    const parsed = parseCardDisplay(cardLabel);
    const classes = ["playing-card", parsed.suitClass];
    if (isCompact) {
      classes.push("is-compact");
    }
    if (isLegal) {
      classes.push("is-legal");
    } else if (isDimmed) {
      classes.push("is-dimmed");
    }
    const dataAttr = isLegal && actionId !== undefined ? `data-action-id="${actionId}"` : "";
    const disabledAttr = isLegal ? "" : "disabled";
    return `
      <button
        type="button"
        class="${classes.join(" ")}"
        aria-label="${escapeHtml(parsed.ariaLabel)}"
        ${dataAttr}
        ${disabledAttr}
      >
        <span class="card-corner">
          <span class="card-rank">${escapeHtml(parsed.displayRank)}</span>
          <span class="card-pip-small" aria-hidden="true">${parsed.symbol}</span>
        </span>
        <span class="card-pip-center" aria-hidden="true">${parsed.symbol}</span>
      </button>
    `;
  }

  function renderSeatZone(seat, ts, humanTurn) {
    const zone = dom.seatZones[seat];
    if (!zone || !ts) {
      return;
    }

    const isActivePhysical = ts.acting_hand === seat && ts.phase !== "terminal";
    zone.classList.toggle("is-active-turn", isActivePhysical);

    const isNs = seat === "NORTH" || seat === "SOUTH";
    const isVul = isNs ? ts.ns_vulnerable : ts.ew_vulnerable;
    const vulBadge = isVul
      ? '<span class="badge-tag badge-vul">VUL</span>'
      : '<span class="badge-tag badge-nv">NV</span>';
    const dealerBadge =
      ts.dealer === seat ? '<span class="badge-tag badge-role-accent">Dealer</span>' : "";
    const roleLabel = seatRoleDescription(seat, ts);
    const hcpVal = ts.hand_hcp ? ts.hand_hcp[seat] : null;
    const hcpBadge =
      typeof hcpVal === "number"
        ? `<span class="badge-tag badge-hcp">${hcpVal} HCP</span>`
        : "";

    const visibleCards = ts.hands ? ts.hands[seat] : null;
    const initialCards =
      ts.phase === "terminal" && ts.initial_hands && ts.initial_hands[seat]
        ? ts.initial_hands[seat]
        : null;
    const cardsToRender =
      ts.phase === "terminal" && initialCards ? initialCards : visibleCards;

    const legalByLabel = new Map();
    if (humanTurn && ts.phase === "play" && ts.acting_hand === seat) {
      for (const act of ts.legal_actions || []) {
        legalByLabel.set(act.label, act.action_id);
      }
    }

    const isSideSeat = seat === "EAST" || seat === "WEST";
    let bodyHtml = "";

    if (Array.isArray(cardsToRender)) {
      if (cardsToRender.length === 0) {
        bodyHtml = '<div class="empty-hand-note">0 cards remaining</div>';
      } else {
        const cardsHtml = cardsToRender
          .map((lbl) => {
            const actionId = legalByLabel.get(lbl);
            const isLegal = actionId !== undefined;
            const isDimmed = legalByLabel.size > 0 && !isLegal;
            return renderCardButtonHtml(lbl, {
              actionId,
              isLegal,
              isDimmed,
              isCompact: isSideSeat,
            });
          })
          .join("");
        bodyHtml = `<div class="hand-cards-fan">${cardsHtml}</div>`;
      }
    } else {
      const remCount = ts.remaining_counts ? ts.remaining_counts[seat] ?? 13 : 13;
      const stackTiles = Math.min(5, Math.max(1, Math.ceil(remCount / 3)));
      let tilesHtml = "";
      for (let i = 0; i < stackTiles; i += 1) {
        tilesHtml += '<span class="card-back-tile" aria-hidden="true"></span>';
      }
      bodyHtml = `
        <div class="card-back-stack">
          <span class="card-back-visuals">${tilesHtml}</span>
          <span class="card-back-count">${remCount} card${remCount === 1 ? "" : "s"}</span>
        </div>
      `;
    }

    zone.innerHTML = `
      <div class="seat-header">
        <div class="seat-identity">
          <span class="seat-compass-letter">${seat[0]}</span>
          <span class="seat-name-text">${seat}</span>
          <span class="badge-tag">${escapeHtml(roleLabel)}</span>
        </div>
        <div class="seat-badges">
          ${dealerBadge}
          ${vulBadge}
          ${hcpBadge}
        </div>
      </div>
      ${bodyHtml}
    `;
  }

  function renderTrickArena(ts) {
    if (!ts) {
      return;
    }

    const completed = ts.completed_tricks || [];
    const lastCompleted = completed.length > 0 ? completed[completed.length - 1] : null;
    const showCompletedOnFelt =
      lastCompleted &&
      (state.inspectingLastTrick || (ts.phase === "play" && (ts.current_trick || []).length === 0));

    const activeCards = showCompletedOnFelt ? lastCompleted.cards : ts.current_trick || [];
    const winnerSeat = showCompletedOnFelt ? lastCompleted.winner : null;

    let captionLeft = "Auction Phase";
    let captionRight = `Seed ${ts.tournament_seed}`;
    if (ts.phase === "play") {
      if (showCompletedOnFelt) {
        captionLeft = `Last Trick #${lastCompleted.trick_number} · Won by ${lastCompleted.winner}`;
        captionRight =
          state.inspectingLastTrick && (ts.current_trick || []).length > 0
            ? "Inspecting previous trick"
            : `Next lead: ${ts.acting_hand || "—"}`;
      } else {
        captionLeft = `Trick #${completed.length + 1}`;
        captionRight = `Turn: ${ts.acting_hand || ts.current_player || "—"}`;
      }
    } else if (ts.phase === "terminal") {
      captionLeft = "Board Complete";
      captionRight = `${completed.length} tricks played`;
    }

    const bySeat = {};
    for (const item of activeCards) {
      bySeat[item.seat] = item;
    }

    const renderSlot = (seat, slotClass) => {
      const entry = bySeat[seat];
      if (!entry) {
        return `
          <div class="trick-slot ${slotClass}">
            <div class="trick-slot-placeholder">${seat[0]}</div>
          </div>
        `;
      }
      const parsed = parseCardDisplay(entry.card);
      const winClass = winnerSeat === seat ? "is-trick-winner" : "";
      return `
        <div class="trick-slot ${slotClass}">
          <div class="trick-card-played ${parsed.suitClass} ${winClass}" title="${seat}: ${escapeHtml(parsed.ariaLabel)}">
            <span class="card-corner">
              <span class="card-rank">${escapeHtml(parsed.displayRank)}</span>
              <span class="card-pip-small">${parsed.symbol}</span>
            </span>
            <span class="card-pip-center">${parsed.symbol}</span>
          </div>
        </div>
      `;
    };

    let terminalBannerHtml = "";
    if (ts.phase === "terminal" && ts.terminal_result) {
      const tr = ts.terminal_result;
      const nsSign = tr.ns_score >= 0 ? `+${tr.ns_score}` : String(tr.ns_score);
      const ewSign = tr.ew_score >= 0 ? `+${tr.ew_score}` : String(tr.ew_score);
      terminalBannerHtml = `
        <div class="terminal-banner">
          <div>
            <div class="terminal-banner-score">
              ${escapeHtml(tr.contract)} · N/S ${nsSign} (E/W ${ewSign})
            </div>
            <div class="terminal-banner-sub">
              Tricks: N/S ${tr.ns_tricks} — E/W ${tr.ew_tricks} · All 4 dealt hands revealed
            </div>
          </div>
          <button type="button" id="next-board-inline-btn" class="btn btn-brass">
            Next Board (#${(ts.board_number % 32) + 1})
          </button>
        </div>
      `;
    }

    dom.trickArena.innerHTML = `
      <div class="trick-arena-caption">
        <span>${escapeHtml(captionLeft)}</span>
        <span>${escapeHtml(captionRight)}</span>
      </div>
      <div class="trick-slots-cross">
        ${renderSlot("NORTH", "trick-slot-north")}
        ${renderSlot("WEST", "trick-slot-west")}
        ${renderSlot("EAST", "trick-slot-east")}
        ${renderSlot("SOUTH", "trick-slot-south")}
      </div>
      ${terminalBannerHtml}
    `;

    const nextBoardBtn = document.getElementById("next-board-inline-btn");
    if (nextBoardBtn) {
      nextBoardBtn.addEventListener("click", () => {
        const nextNum = (state.boardNumber % 32) + 1;
        dom.boardNumberInput.value = String(nextNum);
        startNewDeal({ boardNumber: nextNum, seed: state.tournamentSeed });
      });
    }
  }

  function renderAuctionHistory(ts) {
    if (!ts) {
      return;
    }
    dom.auctionDealerNote.textContent = `Dealer: ${ts.dealer}`;
    const rows = buildAuctionRows(ts.dealer, ts.auction_calls || []);

    const bodyRowsHtml =
      rows.length === 0
        ? '<tr><td colspan="4" class="auction-cell-pad">No calls yet</td></tr>'
        : rows
            .map((row) => {
              const cellsHtml = row
                .map((cell) => {
                  if (!cell) {
                    return '<td class="auction-cell-pad">—</td>';
                  }
                  const fmt = formatCallDisplay(cell.label);
                  if (fmt.suitClass) {
                    return `<td>${escapeHtml(fmt.level)}<span class="${fmt.suitClass}">${fmt.symbol}</span></td>`;
                  }
                  return `<td>${escapeHtml(fmt.display)}</td>`;
                })
                .join("");
              return `<tr>${cellsHtml}</tr>`;
            })
            .join("");

    dom.auctionHistoryGrid.innerHTML = `
      <table class="auction-table" aria-label="4-Column Auction Call History">
        <thead>
          <tr>
            <th scope="col">West</th>
            <th scope="col">North</th>
            <th scope="col">East</th>
            <th scope="col">South</th>
          </tr>
        </thead>
        <tbody>${bodyRowsHtml}</tbody>
      </table>
    `;
  }

  function renderBiddingBoxAndTricksPanel(ts, humanTurn) {
    if (!ts) {
      return;
    }

    const inAuction = ts.phase === "auction";
    dom.biddingBox.hidden = !inAuction;
    dom.playTricksPanel.hidden = inAuction;
    dom.auctionTrayWrapper.classList.toggle("is-play-phase", !inAuction);

    if (inAuction) {
      const legalIds = new Set(
        humanTurn && !state.isBusy ? (ts.legal_actions || []).map((a) => a.action_id) : []
      );

      const specialBtn = (actionId, text, extraClass) => {
        const legal = legalIds.has(actionId);
        return `
          <button
            type="button"
            class="bid-call-btn ${extraClass} ${legal ? "is-legal" : ""}"
            data-bid-action-id="${actionId}"
            ${legal ? "" : "disabled"}
          >
            ${text}
          </button>
        `;
      };

      const denoms = ["C", "D", "H", "S", "NT"];
      let gridHtml = "";
      for (let level = 1; level <= 7; level += 1) {
        for (let dIdx = 0; dIdx < denoms.length; dIdx += 1) {
          const actionId = 55 + (level - 1) * 5 + dIdx;
          const denom = denoms[dIdx];
          const legal = legalIds.has(actionId);
          const meta = SUIT_META[denom];
          const inner = meta
            ? `${level}<span class="${meta.suitClass}">${meta.symbol}</span>`
            : `${level}NT`;
          gridHtml += `
            <button
              type="button"
              class="bid-call-btn ${legal ? "is-legal" : ""}"
              data-bid-action-id="${actionId}"
              ${legal ? "" : "disabled"}
            >
              ${inner}
            </button>
          `;
        }
      }

      const hint = humanTurn
        ? "Your call (South)"
        : state.tableMode === "spectator_tenace"
          ? "Spectator mode"
          : `Waiting for ${ts.current_player || "AI"}`;

      dom.biddingBox.innerHTML = `
        <div class="section-heading-row">
          <h2 class="section-heading">WBF Bidding Box</h2>
          <span class="section-subnote">${escapeHtml(hint)}</span>
        </div>
        <div class="bidding-special-calls">
          ${specialBtn(52, "Pass", "bid-call-pass")}
          ${specialBtn(53, "Dbl (X)", "bid-call-dbl")}
          ${specialBtn(54, "RDbl (XX)", "bid-call-rdbl")}
        </div>
        <div class="bidding-levels-grid">${gridHtml}</div>
      `;
    } else {
      const completed = ts.completed_tricks || [];
      dom.showLastTrickBtn.disabled = completed.length === 0;
      dom.showLastTrickBtn.textContent = state.inspectingLastTrick
        ? "Return to Current Trick"
        : "Inspect Last Trick on Felt";

      if (completed.length === 0) {
        dom.trickHistoryList.innerHTML =
          '<div class="empty-hand-note">Completed tricks will appear here once Trick #1 finishes.</div>';
      } else {
        dom.trickHistoryList.innerHTML = completed
          .slice()
          .reverse()
          .map((tr) => {
            const cardsInline = tr.cards
              .map((c) => {
                const p = parseCardDisplay(c.card);
                return `<span>${c.seat[0]}:${escapeHtml(p.displayRank)}<span class="${p.suitClass}">${p.symbol}</span></span>`;
              })
              .join(" · ");
            return `
              <div class="trick-history-row">
                <span>#${tr.trick_number} (${tr.leader[0]} led)</span>
                <span>${cardsInline}</span>
                <strong>→ ${tr.winner}</strong>
              </div>
            `;
          })
          .join("");
      }
    }
  }

  function renderPlaquesAndTurnBar(ts, humanTurn) {
    if (!ts) {
      return;
    }

    const vulLabel =
      ts.ns_vulnerable && ts.ew_vulnerable
        ? "Both"
        : ts.ns_vulnerable
          ? "N-S"
          : ts.ew_vulnerable
            ? "E-W"
            : "None";

    dom.plaqueBoardTitle.textContent = `Board #${ts.board_number}`;
    dom.plaqueDealerVul.textContent = `Dealer: ${ts.dealer[0]} · Vul: ${vulLabel}`;
    dom.contractPlaqueValue.innerHTML = formatContractHtml(ts.contract);
    dom.contractPlaqueTricks.textContent = `N/S ${ts.ns_tricks} — E/W ${ts.ew_tricks}`;

    dom.turnPhasePill.textContent = `Phase: ${ts.phase.toUpperCase()}`;
    dom.turnLatencyPill.textContent =
      state.lastStepLatencyMs !== null ? `Last Step: ${state.lastStepLatencyMs} ms` : "Last Step: —";

    dom.turnSpinner.hidden = !state.isBusy;
    const isDummyControl =
      humanTurn && ts.phase === "play" && ts.current_player === "SOUTH" && ts.acting_hand === "NORTH";
    dom.turnStatusGroup.classList.toggle("is-human-dummy-prompt", isDummyControl);

    if (state.isBusy) {
      const seat = ts.current_player || "AI";
      const botKind = resolveBotForSeat(state.tableMode, seat);
      dom.turnStatusText.textContent =
        botKind === "baseline"
          ? `${seat} (Fast Baseline) selecting action…`
          : `${seat} (Tenace AI) evaluating options…`;
    } else if (ts.phase === "terminal") {
      dom.turnStatusText.textContent = ts.terminal_result
        ? `Hand complete — ${ts.terminal_result.contract} (N/S ${ts.terminal_result.ns_score >= 0 ? "+" : ""}${ts.terminal_result.ns_score})`
        : "Hand complete";
    } else if (isDummyControl) {
      dom.turnStatusText.textContent =
        "Your turn (Declarer) — play a highlighted card from Dummy (North)";
    } else if (humanTurn) {
      dom.turnStatusText.textContent =
        ts.phase === "auction"
          ? "Your turn (South) — select a call from the WBF Bidding Box"
          : "Your turn (South) — click a legal card in your hand";
    } else {
      dom.turnStatusText.textContent = state.autoPlayAi
        ? `AI turn: ${ts.current_player}${ts.acting_hand !== ts.current_player ? ` (playing ${ts.acting_hand})` : ""}`
        : `Paused on ${ts.current_player} — click "Step AI" or enable Auto-Play`;
    }

    dom.undoBtn.disabled = state.isBusy || state.timeline.length === 0;
    dom.stepAiBtn.disabled = state.isBusy || ts.phase === "terminal" || humanTurn;
  }

  function renderTelemetryAndStrategyFeed() {
    dom.statStrategistCalls.textContent = String(state.telemetryTotals.strategist_calls);
    dom.statEvaluatorCalls.textContent = String(state.telemetryTotals.evaluator_calls);
    dom.statForcedMoves.textContent = String(state.telemetryTotals.forced_moves);
    dom.statLastMs.textContent =
      state.lastStepLatencyMs !== null ? `${state.lastStepLatencyMs}ms` : "—";

    const visibleEntries = state.strategyFeed.filter((entry) => {
      if (state.hideEwThoughts && (entry.seat === "EAST" || entry.seat === "WEST")) {
        return false;
      }
      return true;
    });

    if (visibleEntries.length === 0) {
      dom.strategyFeed.innerHTML = `
        <p class="empty-hand-note">
          ${
            state.strategyFeed.length > 0 && state.hideEwThoughts
              ? "E/W AI thoughts are currently hidden. Uncheck 'Hide E/W' to inspect defender plans."
              : "Layer-1 StrategyState plans, target/threat assessments, and HCP constraints appear here as Tenace seats act."
          }
        </p>
      `;
      return;
    }

    dom.strategyFeed.innerHTML = visibleEntries
      .slice()
      .reverse()
      .map((entry) => {
        const strat = entry.strategy;
        const actionFmt =
          entry.phase === "auction"
            ? formatCallDisplay(entry.actionLabel).display
            : `${parseCardDisplay(entry.actionLabel).displayRank}${parseCardDisplay(entry.actionLabel).symbol}`;

        if (!strat) {
          return `
            <article class="strategy-entry">
              <div class="strategy-entry-header">
                <span>[${escapeHtml(entry.seat)} · ${escapeHtml(entry.botLabel)}]</span>
                <span>→ ${escapeHtml(actionFmt)} (${entry.latencyMs}ms)</span>
              </div>
              <p class="strategy-target">0-LLM tactical/forced action (${entry.forcedMoves ? "forced move" : "baseline"}).</p>
            </article>
          `;
        }

        const assumptions = Array.isArray(strat.assumptions) ? strat.assumptions : [];
        const assumptionsHtml =
          assumptions.length > 0
            ? `<ul class="strategy-assumptions">${assumptions
                .map(
                  (a) =>
                    `<li>${escapeHtml(a.claim)} <strong>(${Math.round((a.confidence ?? 0.6) * 100)}%)</strong></li>`
                )
                .join("")}</ul>`
            : "";

        const rawConstraints =
          strat.seat_constraints && typeof strat.seat_constraints === "object"
            ? Object.entries(strat.seat_constraints)
            : [];
        const constraintsHtml =
          rawConstraints.length > 0
            ? `<ul class="strategy-constraints">${rawConstraints
                .map(([key, c]) => {
                  const seatName =
                    SEAT_INDEX_NAMES[c && c.seat !== undefined ? c.seat : key] ||
                    SEAT_INDEX_NAMES[key] ||
                    String(key);
                  const minHcp = c.min_hcp ?? 0;
                  const maxHcp = c.max_hcp ?? 37;
                  const suitParts = [];
                  if (c.min_suit_lengths && typeof c.min_suit_lengths === "object") {
                    for (const [sKey, sMin] of Object.entries(c.min_suit_lengths)) {
                      const sym = SUIT_ENUM_SYMBOLS[sKey] || sKey;
                      suitParts.push(`${sMin}+${sym}`);
                    }
                  }
                  const suitNote = suitParts.length > 0 ? ` · ${suitParts.join(", ")}` : "";
                  return `<li><strong>${escapeHtml(seatName)}</strong>: ${minHcp}–${maxHcp} HCP${escapeHtml(suitNote)}</li>`;
                })
                .join("")}</ul>`
            : "";

        const replanBadge = entry.replanned ? " · NEW PLAN" : " · cached";
        return `
          <article class="strategy-entry">
            <div class="strategy-entry-header">
              <span>[${escapeHtml(entry.seat)} · rev ${strat.revision ?? 0} · ${escapeHtml(strat.phase)}${replanBadge}]</span>
              <span>→ ${escapeHtml(actionFmt)} (${entry.latencyMs}ms)</span>
            </div>
            <p class="strategy-plan">${escapeHtml(strat.plan_summary)}</p>
            ${
              strat.target_or_threat
                ? `<p class="strategy-target">Target/Threat: ${escapeHtml(strat.target_or_threat)}</p>`
                : ""
            }
            ${assumptionsHtml}
            ${constraintsHtml}
          </article>
        `;
      })
      .join("");
  }

  function renderAll() {
    const ts = state.tableState;
    const humanTurn = isHumanDecisionTurn(state.tableMode, ts);
    if (ts) {
      for (const seat of ["NORTH", "WEST", "EAST", "SOUTH"]) {
        renderSeatZone(seat, ts, humanTurn && !state.isBusy);
      }
      renderTrickArena(ts);
      renderAuctionHistory(ts);
      renderBiddingBoxAndTricksPanel(ts, humanTurn);
      renderPlaquesAndTurnBar(ts, humanTurn);
    }
    renderTelemetryAndStrategyFeed();
  }

  /* ========================================================================
     Table State Sync & AI Turn Loop
     ======================================================================== */

  async function refreshTableState(epoch = state.sessionEpoch) {
    const controller = new AbortController();
    state.activeAbortController = controller;

    const actionHistory = state.timeline.map((item) => item.action);
    const ts = await fetchJson("/v1/table/state", {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify({
        tournament_seed: state.tournamentSeed,
        board_number: state.boardNumber,
        action_history: actionHistory,
        viewer_seat: "SOUTH",
        reveal_all: Boolean(state.revealHands),
      }),
    });

    if (epoch !== state.sessionEpoch) {
      return null;
    }

    const prevCompleted = state.prevCompletedTrickCount;
    const nowCompleted = (ts.completed_tricks || []).length;
    const trickJustCompleted = nowCompleted > prevCompleted && (ts.current_trick || []).length === 0;
    state.prevCompletedTrickCount = nowCompleted;
    if ((ts.current_trick || []).length > 0) {
      state.inspectingLastTrick = false;
    }

    state.tableState = ts;
    renderAll();

    if (
      state.activeTab === "play" &&
      state.autoPlayAi &&
      ts.phase !== "terminal" &&
      !isHumanDecisionTurn(state.tableMode, ts)
    ) {
      const delayMs = trickJustCompleted ? 680 : 320;
      state.autoPlayTimerId = window.setTimeout(() => {
        state.autoPlayTimerId = null;
        if (epoch === state.sessionEpoch) {
          executeAiStep();
        }
      }, delayMs);
    }

    return ts;
  }

  async function startNewDeal({ boardNumber, seed } = {}) {
    cancelPendingAsyncWork();
    clearError();

    state.boardNumber = clampInt(
      boardNumber !== undefined ? boardNumber : dom.boardNumberInput.value,
      1,
      32,
      1
    );
    state.tournamentSeed = clampSeed(
      seed !== undefined ? seed : dom.seedInput.value,
      42
    );
    dom.boardNumberInput.value = String(state.boardNumber);
    dom.seedInput.value = String(state.tournamentSeed);

    state.timeline = [];
    state.priorStrategies = { NORTH: null, EAST: null, SOUTH: null, WEST: null };
    state.telemetryTotals = { strategist_calls: 0, evaluator_calls: 0, forced_moves: 0 };
    state.lastStepLatencyMs = null;
    state.strategyFeed = [];
    state.prevCompletedTrickCount = 0;
    state.inspectingLastTrick = false;

    const myEpoch = state.sessionEpoch;
    try {
      await refreshTableState(myEpoch);
    } catch (err) {
      if (myEpoch === state.sessionEpoch && err.name !== "AbortError") {
        showError(`Failed to load board state: ${err.message}`);
      }
    }
  }

  async function submitHumanAction(actionId) {
    const ts = state.tableState;
    if (state.isBusy || !isHumanDecisionTurn(state.tableMode, ts)) {
      return;
    }
    const legalMatch = (ts.legal_actions || []).find((a) => a.action_id === actionId);
    if (!legalMatch) {
      return;
    }

    cancelPendingAsyncWork();
    clearError();
    state.inspectingLastTrick = false;

    state.timeline.push({
      action: actionId,
      actorPlayer: "SOUTH",
      priorStrategiesBefore: JSON.parse(JSON.stringify(state.priorStrategies)),
      telemetryDelta: { strategist_calls: 0, evaluator_calls: 0, forced_moves: 0 },
      feedEntryId: null,
    });

    const myEpoch = state.sessionEpoch;
    try {
      await refreshTableState(myEpoch);
    } catch (err) {
      if (myEpoch === state.sessionEpoch && err.name !== "AbortError") {
        state.timeline.pop();
        showError(`Illegal or failed action: ${err.message}`);
        renderAll();
      }
    }
  }

  async function executeAiStep() {
    const ts = state.tableState;
    if (
      state.isBusy ||
      !ts ||
      ts.phase === "terminal" ||
      isHumanDecisionTurn(state.tableMode, ts)
    ) {
      return;
    }

    if (state.autoPlayTimerId !== null) {
      window.clearTimeout(state.autoPlayTimerId);
      state.autoPlayTimerId = null;
    }
    clearError();

    const myEpoch = state.sessionEpoch;
    const seat = ts.current_player;
    const bot = resolveBotForSeat(state.tableMode, seat);
    const priorBefore = JSON.parse(JSON.stringify(state.priorStrategies));
    const seatPrior = state.priorStrategies[seat];

    // Guard against sending a stale play prior_strategy into an auction phase
    const validPrior =
      seatPrior && !(ts.phase === "auction" && seatPrior.phase === "play")
        ? seatPrior
        : null;

    const ddsWorlds = clampInt(dom.ddsWorldsInput.value, 1, 64, state.serverDdsWorlds);
    dom.ddsWorldsInput.value = String(ddsWorlds);

    const profileOverride =
      bot === "tenace"
        ? buildProfilePayload(readTableUiProfile(), state.serverProfile, false)
        : null;

    const stepSeed = clampSeed(
      (state.tournamentSeed * 31 + state.boardNumber * 17 + state.timeline.length) % 1000000000,
      42
    );

    const requestBody = {
      seat,
      board_number: state.boardNumber,
      tournament_seed: state.tournamentSeed,
      action_history: state.timeline.map((item) => item.action),
      prior_strategy: validPrior,
      dds_worlds: ddsWorlds,
      seed: stepSeed,
      bot,
    };
    if (profileOverride) {
      requestBody.profile = profileOverride;
    }

    const controller = new AbortController();
    state.activeAbortController = controller;
    state.isBusy = true;
    renderAll();

    const t0 = window.performance.now();
    try {
      const stepResp = await fetchJson("/v1/step", {
        method: "POST",
        signal: controller.signal,
        body: JSON.stringify(requestBody),
      });

      if (myEpoch !== state.sessionEpoch) {
        return;
      }

      const elapsedMs = Math.max(1, Math.round(window.performance.now() - t0));
      state.lastStepLatencyMs = elapsedMs;

      const tel = stepResp.telemetry || {
        strategist_calls: 0,
        evaluator_calls: 0,
        forced_moves: 0,
      };
      state.telemetryTotals.strategist_calls += tel.strategist_calls || 0;
      state.telemetryTotals.evaluator_calls += tel.evaluator_calls || 0;
      state.telemetryTotals.forced_moves += tel.forced_moves || 0;

      if (stepResp.strategy) {
        state.priorStrategies[seat] = stepResp.strategy;
      } else if (bot === "baseline" && ts.phase === "play" && validPrior && validPrior.phase === "auction") {
        state.priorStrategies[seat] = null;
      }

      const feedEntryId = state.nextFeedId;
      state.nextFeedId += 1;
      state.strategyFeed.push({
        id: feedEntryId,
        seat,
        phase: stepResp.phase,
        actionLabel: stepResp.action_label,
        botLabel: bot === "baseline" ? "Baseline" : "Tenace",
        strategy: stepResp.strategy,
        replanned: (tel.strategist_calls || 0) > 0,
        forcedMoves: tel.forced_moves || 0,
        latencyMs: elapsedMs,
      });

      state.timeline.push({
        action: stepResp.action_id,
        actorPlayer: seat,
        priorStrategiesBefore: priorBefore,
        telemetryDelta: tel,
        feedEntryId,
      });

      state.isBusy = false;
      await refreshTableState(myEpoch);
    } catch (err) {
      if (myEpoch !== state.sessionEpoch || err.name === "AbortError") {
        return;
      }
      state.isBusy = false;
      showError(`AI step failed (${seat}): ${err.message}`);
      renderAll();
    }
  }

  async function handleUndo() {
    if (state.timeline.length === 0) {
      return;
    }
    cancelPendingAsyncWork();
    clearError();
    state.inspectingLastTrick = false;

    const targetLen = computeUndoTargetLength(state.tableMode, state.timeline);
    while (state.timeline.length > targetLen) {
      const popped = state.timeline.pop();
      if (!popped) {
        break;
      }
      state.priorStrategies = popped.priorStrategiesBefore;
      state.telemetryTotals.strategist_calls = Math.max(
        0,
        state.telemetryTotals.strategist_calls - (popped.telemetryDelta.strategist_calls || 0)
      );
      state.telemetryTotals.evaluator_calls = Math.max(
        0,
        state.telemetryTotals.evaluator_calls - (popped.telemetryDelta.evaluator_calls || 0)
      );
      state.telemetryTotals.forced_moves = Math.max(
        0,
        state.telemetryTotals.forced_moves - (popped.telemetryDelta.forced_moves || 0)
      );
      if (popped.feedEntryId !== null) {
        state.strategyFeed = state.strategyFeed.filter((f) => f.id !== popped.feedEntryId);
      }
    }

    const myEpoch = state.sessionEpoch;
    try {
      await refreshTableState(myEpoch);
    } catch (err) {
      if (myEpoch === state.sessionEpoch && err.name !== "AbortError") {
        showError(`Undo sync failed: ${err.message}`);
      }
    }
  }

  /* ========================================================================
     Tab 2: Duplicate IMP Match Lab Controller
     ======================================================================== */

  function renderLabMatchResults(matchResp, elapsedSec) {
    const summary = matchResp.summary;
    const impsA = summary.total_imps_team_a;
    const impsB = summary.total_imps_team_b;
    const signA = impsA >= 0 ? `+${impsA}` : String(impsA);
    const signB = impsB >= 0 ? `+${impsB}` : String(impsB);

    const outcomeLabel =
      impsA > impsB
        ? `${matchResp.team_a_name} Wins`
        : impsB > impsA
          ? `${matchResp.team_b_name} Wins`
          : "Exact Tie";

    const telEntries = Object.entries(summary.telemetry || {})
      .map(([k, v]) => `<span class="badge-tag">${escapeHtml(k)}: ${v}</span>`)
      .join(" ");

    const boardsHtml = (summary.boards || [])
      .map((b) => {
        const swing = b.imps_team_a;
        const swingText =
          swing === 0
            ? "Push (0 IMPs)"
            : swing > 0
              ? `${escapeHtml(matchResp.team_a_name)} +${swing} IMPs`
              : `${escapeHtml(matchResp.team_b_name)} +${Math.abs(swing)} IMPs`;

        const renderRoomBox = (title, room, fallbackScore) => {
          if (!room) {
            return `
              <div class="lab-room-box">
                <strong>${escapeHtml(title)}</strong>
                <div>N/S Score: ${fallbackScore}</div>
              </div>
            `;
          }
          const traces = Object.entries(room.strategy_traces || {})
            .map(
              ([seatName, list]) =>
                `<div><strong>${escapeHtml(seatName)}:</strong> ${escapeHtml((list || []).join(" → "))}</div>`
            )
            .join("");
          return `
            <div class="lab-room-box">
              <div class="section-heading-row">
                <strong>${escapeHtml(title)}</strong>
                <span class="badge-tag badge-role-accent">${escapeHtml(room.contract)}</span>
              </div>
              <div>N/S Score: <strong>${room.ns_score >= 0 ? `+${room.ns_score}` : room.ns_score}</strong></div>
              <div style="margin-top:0.28rem;color:var(--text-sand);">Auction: <code>${escapeHtml(room.auction || "—")}</code></div>
              ${
                traces
                  ? `<div style="margin-top:0.4rem;font-size:0.73rem;color:var(--text-muted);">${traces}</div>`
                  : ""
              }
            </div>
          `;
        };

        return `
          <article class="lab-board-card">
            <div class="section-heading-row">
              <h3 class="section-heading">Board #${b.board_number}</h3>
              <span class="badge-tag badge-role-accent">${swingText}</span>
            </div>
            <div class="lab-rooms-split">
              ${renderRoomBox(`Open Room (${matchResp.team_a_name} N/S)`, b.open_room, b.open_room_ns_score)}
              ${renderRoomBox(`Closed Room (${matchResp.team_b_name} N/S)`, b.closed_room, b.closed_room_ns_score)}
            </div>
          </article>
        `;
      })
      .join("");

    dom.labResults.innerHTML = `
      <div class="lab-scoreboard">
        <div>
          <div class="section-subnote">Symmetric Two-Room WBF Duplicate Result (${elapsedSec}s)</div>
          <div class="lab-score-numbers">
            ${escapeHtml(matchResp.team_a_name)} ${signA} IMPs — ${escapeHtml(matchResp.team_b_name)} ${signB} IMPs
          </div>
          <div style="font-size:0.78rem;color:var(--text-sand);margin-top:0.25rem;">
            A: <code>${escapeHtml(matchResp.team_a_profile)}</code>
            ${matchResp.team_b_profile ? ` · B: <code>${escapeHtml(matchResp.team_b_profile)}</code>` : ""}
          </div>
        </div>
        <span class="badge-tag badge-role-accent" style="font-size:0.86rem;padding:0.35rem 0.7rem;">
          ${escapeHtml(outcomeLabel)}
        </span>
      </div>

      <div style="display:flex;flex-wrap:wrap;gap:0.35rem;margin-bottom:0.85rem;">
        ${telEntries}
      </div>

      <div class="lab-board-cards">${boardsHtml}</div>

      <details class="raw-report-details">
        <summary>View Full Formatted Match Scorecard Text</summary>
        <pre class="raw-report-pre">${escapeHtml(matchResp.report)}</pre>
      </details>
    `;
  }

  async function handleRunLabMatch(event) {
    event.preventDefault();

    const boards = clampInt(dom.labBoardsInput.value, 1, 8, 2);
    const seed = clampSeed(dom.labSeedInput.value, 42);
    const ddsWorlds = clampInt(dom.labDdsWorldsInput.value, 1, 64, 4);
    const opponent = dom.labOpponentSelect.value === "tenace" ? "tenace" : "baseline";
    const doubleDummy = Boolean(dom.labDoubleDummy.checked);

    dom.labBoardsInput.value = String(boards);
    dom.labSeedInput.value = String(seed);
    dom.labDdsWorldsInput.value = String(ddsWorlds);

    const teamAOverride = buildProfilePayload(
      {
        mode: dom.labTeamAMode.value,
        strategist_model: dom.labTeamAStrategist.value,
        evaluator_backend: dom.labTeamABackend.value,
        evaluator_model: dom.labTeamAEvaluator.value,
      },
      state.serverProfile,
      false
    );

    const payload = {
      boards,
      seed,
      dds_worlds: ddsWorlds,
      double_dummy: doubleDummy,
      opponent,
    };
    if (teamAOverride) {
      payload.team_a_profile = teamAOverride;
    }
    if (opponent === "tenace") {
      payload.team_b_profile = buildProfilePayload(
        {
          mode: dom.labTeamBMode.value,
          strategist_model: dom.labTeamBStrategist.value,
          evaluator_backend: dom.labTeamBBackend.value,
          evaluator_model: dom.labTeamBEvaluator.value,
        },
        state.serverProfile,
        true
      );
    }

    dom.labRunBtn.disabled = true;
    dom.labRunBtn.textContent = "Running Duplicate Match…";
    dom.labResults.innerHTML = `
      <p class="empty-hand-note">
        Playing ${boards} duplicate board${boards === 1 ? "" : "s"} across Open &amp; Closed Rooms…
      </p>
    `;

    const t0 = window.performance.now();
    try {
      const matchResp = await fetchJson("/v1/match", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const elapsedSec = ((window.performance.now() - t0) / 1000).toFixed(1);
      renderLabMatchResults(matchResp, elapsedSec);
    } catch (err) {
      dom.labResults.innerHTML = `
        <div class="error-banner" role="alert">
          <span>Match Lab error: ${escapeHtml(err.message)}</span>
        </div>
      `;
    } finally {
      dom.labRunBtn.disabled = false;
      dom.labRunBtn.textContent = "Run Duplicate IMP Match";
    }
  }

  /* ========================================================================
     Event Wiring & Initialization
     ======================================================================== */

  function switchTab(targetTab) {
    state.activeTab = targetTab;
    const isPlay = targetTab === "play";
    dom.viewPlay.hidden = !isPlay;
    dom.viewLab.hidden = isPlay;
    dom.tabPlay.classList.toggle("is-active", isPlay);
    dom.tabLab.classList.toggle("is-active", !isPlay);
    dom.tabPlay.setAttribute("aria-pressed", String(isPlay));
    dom.tabLab.setAttribute("aria-pressed", String(!isPlay));

    if (!isPlay && state.autoPlayTimerId !== null) {
      window.clearTimeout(state.autoPlayTimerId);
      state.autoPlayTimerId = null;
    } else if (
      isPlay &&
      state.autoPlayAi &&
      state.tableState &&
      state.tableState.phase !== "terminal" &&
      !isHumanDecisionTurn(state.tableMode, state.tableState) &&
      !state.isBusy
    ) {
      executeAiStep();
    }
  }

  function wireEvents() {
    dom.tabPlay.addEventListener("click", () => switchTab("play"));
    dom.tabLab.addEventListener("click", () => switchTab("lab"));

    dom.tableModeSelect.addEventListener("change", () => {
      cancelPendingAsyncWork();
      clearError();
      state.tableMode = dom.tableModeSelect.value;
      renderAll();
      if (
        state.autoPlayAi &&
        state.tableState &&
        state.tableState.phase !== "terminal" &&
        !isHumanDecisionTurn(state.tableMode, state.tableState)
      ) {
        executeAiStep();
      }
    });

    dom.newDealBtn.addEventListener("click", () => {
      startNewDeal();
    });

    dom.randomDealBtn.addEventListener("click", () => {
      const nextSeed = randomSafeSeed();
      const nextBoard = Math.floor(Math.random() * 16) + 1;
      dom.seedInput.value = String(nextSeed);
      dom.boardNumberInput.value = String(nextBoard);
      startNewDeal({ boardNumber: nextBoard, seed: nextSeed });
    });

    dom.undoBtn.addEventListener("click", () => {
      handleUndo();
    });

    dom.stepAiBtn.addEventListener("click", () => {
      executeAiStep();
    });

    dom.autoPlayBtn.addEventListener("click", () => {
      state.autoPlayAi = !state.autoPlayAi;
      dom.autoPlayBtn.setAttribute("aria-pressed", String(state.autoPlayAi));
      dom.autoPlayBtn.textContent = state.autoPlayAi ? "Auto-Play AI: On" : "Auto-Play AI: Off";
      if (!state.autoPlayAi && state.autoPlayTimerId !== null) {
        window.clearTimeout(state.autoPlayTimerId);
        state.autoPlayTimerId = null;
      } else if (
        state.autoPlayAi &&
        state.tableState &&
        state.tableState.phase !== "terminal" &&
        !isHumanDecisionTurn(state.tableMode, state.tableState) &&
        !state.isBusy
      ) {
        executeAiStep();
      }
      renderAll();
    });

    dom.revealHandsToggle.addEventListener("change", async () => {
      state.revealHands = Boolean(dom.revealHandsToggle.checked);
      cancelPendingAsyncWork();
      const myEpoch = state.sessionEpoch;
      try {
        await refreshTableState(myEpoch);
      } catch (err) {
        if (myEpoch === state.sessionEpoch && err.name !== "AbortError") {
          showError(`Failed to toggle hand visibility: ${err.message}`);
        }
      }
    });

    dom.fourColorToggle.addEventListener("change", () => {
      state.fourColorDeck = Boolean(dom.fourColorToggle.checked);
      document.body.classList.toggle("classic-deck", !state.fourColorDeck);
    });

    dom.showLastTrickBtn.addEventListener("click", () => {
      state.inspectingLastTrick = !state.inspectingLastTrick;
      renderAll();
    });

    dom.retryStepBtn.addEventListener("click", () => {
      clearError();
      if (state.tableState && !isHumanDecisionTurn(state.tableMode, state.tableState)) {
        executeAiStep();
      } else {
        startNewDeal();
      }
    });

    dom.dismissErrorBtn.addEventListener("click", () => {
      clearError();
    });

    // Delegated click handler for legal playing cards across South & Dummy North
    for (const seat of ["NORTH", "SOUTH", "EAST", "WEST"]) {
      dom.seatZones[seat].addEventListener("click", (ev) => {
        const cardBtn = ev.target.closest("button[data-action-id]");
        if (!cardBtn || cardBtn.disabled) {
          return;
        }
        const actionId = Number.parseInt(cardBtn.getAttribute("data-action-id"), 10);
        if (Number.isInteger(actionId)) {
          submitHumanAction(actionId);
        }
      });
    }

    // Delegated click handler for WBF Bidding Box
    dom.biddingBox.addEventListener("click", (ev) => {
      const bidBtn = ev.target.closest("button[data-bid-action-id]");
      if (!bidBtn || bidBtn.disabled) {
        return;
      }
      const actionId = Number.parseInt(bidBtn.getAttribute("data-bid-action-id"), 10);
      if (Number.isInteger(actionId)) {
        submitHumanAction(actionId);
      }
    });

    // Auto-update default Layer-2 model when switching backend
    dom.evaluatorBackendSelect.addEventListener("change", () => {
      dom.evaluatorModelInput.value = defaultEvaluatorModelForBackend(
        dom.evaluatorBackendSelect.value,
        state.serverProfile
      );
    });

    dom.labTeamABackend.addEventListener("change", () => {
      dom.labTeamAEvaluator.value = defaultEvaluatorModelForBackend(
        dom.labTeamABackend.value,
        state.serverProfile
      );
    });

    dom.labTeamBBackend.addEventListener("change", () => {
      dom.labTeamBEvaluator.value = defaultEvaluatorModelForBackend(
        dom.labTeamBBackend.value,
        null
      );
    });

    dom.resetProfileBtn.addEventListener("click", () => {
      applyServerDefaultsToForms();
    });

    dom.hideEwThoughtsToggle.addEventListener("change", () => {
      state.hideEwThoughts = Boolean(dom.hideEwThoughtsToggle.checked);
      renderTelemetryAndStrategyFeed();
    });

    dom.labOpponentSelect.addEventListener("change", () => {
      dom.labTeamBFieldset.hidden = dom.labOpponentSelect.value !== "tenace";
    });

    dom.labForm.addEventListener("submit", handleRunLabMatch);
  }

  async function initApp() {
    wireEvents();
    try {
      const health = await fetchJson("/health");
      state.serverHealth = health;
      state.serverProfile = health.profile || null;
      state.serverDdsWorlds = clampInt(health.dds_worlds, 1, 64, 8);
      applyServerDefaultsToForms();

      dom.serverDot.classList.remove("is-offline");
      dom.serverText.textContent = `${health.default_profile} · DDS ${health.dds_worlds}`;
    } catch (err) {
      dom.serverDot.classList.add("is-offline");
      dom.serverText.textContent = "Service unreachable";
      showError(`Could not reach /health: ${err.message}`);
    }

    await startNewDeal({ boardNumber: 1, seed: 42 });
  }

  initApp();
})();
