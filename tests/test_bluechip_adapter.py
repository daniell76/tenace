from __future__ import annotations

import inspect
import re
import shlex
import socket
import sys
import threading

import pyspiel
import pytest

from tenace.domain.types import PASS_ACTION, Seat, Suit, card_id
from tenace.harness import DuplicateMatchRunner, FirstLegalPassBaselineBot
import tenace
import tenace.harness
from tenace.harness.bluechip import (
    BlueChipLineTransport,
    BlueChipSeatBot,
    SocketBlueChipController,
    SubprocessTcpBlueChipController,
    format_bluechip_deal_header,
    format_bluechip_hand,
    openspiel_action_to_bluechip,
    parse_bluechip_player_action,
)


class LoopbackBlueChipPeer:
    """Deterministic BlueChip v18 engine running in a background thread over real TCP."""

    def __init__(
        self,
        port: int,
        open_1nt_once: bool = True,
        wbridge5_double_spaces: bool = False,
        pick_highest_card: bool = False,
    ) -> None:
        self.port = port
        self.open_1nt_once = open_1nt_once
        self.wbridge5_double_spaces = wbridge5_double_spaces
        self.pick_highest_card = pick_highest_card
        self.deal_headers: list[str] = []
        self.to_lead_prompts: list[str] = []
        self.event_log: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            self._run_session()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _run_session(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5.0) as sock:
            rfile = sock.makefile("r", encoding="ascii", newline="\r\n")

            def send(msg: str) -> None:
                self.event_log.append(f"SEND: {msg}")
                sock.sendall((msg + "\r\n").encode("ascii"))

            def recv() -> str:
                line = rfile.readline()
                text = line.strip() if line else ""
                self.event_log.append(f"RECV: {text}")
                return text

            send('Connecting "FakeWB5" as ANYPL using protocol version 18')
            seated = recv()  # e.g. North ("FakeWB5") seated
            if not seated:
                return
            seat = seated.split()[0]
            send(f"{seat} ready for teams")
            teams_line = recv()
            if not teams_line:
                return
            if not re.search(r'N/S : "([^"]+)" E/W : "([^"]+)"', teams_line):
                raise AssertionError(f"Unexpected BlueChip teams line: {teams_line!r}")
            send(f"{seat} ready to start")

            seats_order = ["North", "East", "South", "West"]
            assert seat in seats_order, f"Expected Title-Case seat name, got {seat!r}"
            suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}
            rank_chars = "23456789tjqka"
            sp = "  " if self.wbridge5_double_spaces else " "

            while True:
                line = recv()
                if not line:
                    break
                if line.lower() != "start of board":
                    continue
                send(f"{seat} ready for deal")
                deal_hdr = recv()
                if not deal_hdr:
                    break
                self.deal_headers.append(deal_hdr)
                m_dealer = re.search(r"Dealer (North|East|South|West)\.", deal_hdr)
                assert m_dealer is not None, f"Expected Title-Case dealer in {deal_hdr!r}"
                dealer = m_dealer.group(1)

                send(f"{seat} ready for cards")
                cards_line = recv()  # Seat's cards: S ... H ... D ... C ...
                if not cards_line:
                    break
                assert cards_line.startswith(f"{seat}'s cards: S "), cards_line
                own_cards = self._parse_hand_line(cards_line)
                dummy_cards: list[str] = []

                # Auction loop
                turn_idx = seats_order.index(dealer)
                passes = 0
                has_bid = False
                last_bid_denom = "NT"
                first_bidder: dict[tuple[int, str], str] = {}
                winning_side = 0
                declarer = "North"

                disconnected = False
                while True:
                    actor = seats_order[turn_idx]
                    if actor == seat:
                        if not has_bid and self.open_1nt_once:
                            call_msg = f"{seat} bids 1NT"
                            has_bid = True
                            passes = 0
                            last_bid_denom = "NT"
                            winning_side = turn_idx % 2
                            first_bidder.setdefault((winning_side, "NT"), actor)
                            declarer = first_bidder[(winning_side, "NT")]
                        else:
                            call_msg = f"{seat} PASSES"
                            passes += 1
                        send(call_msg)
                    else:
                        send(f"{seat} ready for{sp}{actor}'s bid")
                        obs_line = recv()
                        if not obs_line:
                            disconnected = True
                            break
                        assert obs_line.split()[0] == actor, (
                            f"Expected Title-Case actor {actor!r}, got {obs_line!r}"
                        )
                        if "PASSES" in obs_line:
                            passes += 1
                        elif "bids " in obs_line:
                            bid_tok = obs_line.split("bids ")[1].split()[0]
                            denom = bid_tok[1:]
                            has_bid = True
                            passes = 0
                            last_bid_denom = denom
                            winning_side = turn_idx % 2
                            first_bidder.setdefault((winning_side, denom), actor)
                            declarer = first_bidder[(winning_side, denom)]
                        else:
                            passes = 0

                    turn_idx = (turn_idx + 1) % 4
                    if (not has_bid and passes == 4) or (has_bid and passes == 3):
                        break

                if disconnected:
                    break

                if not has_bid:
                    if not recv():
                        break
                    continue

                dummy_seat = seats_order[(seats_order.index(declarer) + 2) % 4]
                leader_idx = (seats_order.index(declarer) + 1) % 4

                for trick_no in range(1, 14):
                    trick_cards: list[tuple[int, str]] = []
                    led_suit = ""
                    for offset in range(4):
                        p_idx = (leader_idx + offset) % 4
                        p_seat = seats_order[p_idx]
                        is_our_action = (p_seat == seat and seat != dummy_seat) or (
                            p_seat == dummy_seat and seat == declarer
                        )
                        if offset == 0 and is_our_action:
                            to_lead = recv()  # "{Seat|Dummy} to lead"
                            if not to_lead:
                                disconnected = True
                                break
                            self.to_lead_prompts.append(to_lead)

                        if is_our_action:
                            hand_ref = dummy_cards if p_seat == dummy_seat else own_cards
                            chosen = self._pick_legal(
                                hand_ref,
                                led_suit,
                                pick_highest=self.pick_highest_card,
                                ranks=rank_chars,
                            )
                            hand_ref.remove(chosen)
                            actor_label = p_seat
                            send(f"{actor_label} plays {chosen}")
                            card_tok = chosen
                        else:
                            req_target = "Dummy" if p_seat == dummy_seat else p_seat
                            send(f"{seat} ready for{sp}{req_target}'s card to trick {trick_no}")
                            play_line = recv()
                            if not play_line:
                                disconnected = True
                                break
                            assert play_line.split()[0] == p_seat, (
                                f"Expected Title-Case actor {p_seat!r}, got {play_line!r}"
                            )
                            card_tok = play_line.split("plays ")[1].split()[0].lower()
                            if p_seat == dummy_seat and card_tok in dummy_cards:
                                dummy_cards.remove(card_tok)

                        if offset == 0:
                            led_suit = card_tok[1]
                        trick_cards.append((p_idx, card_tok))

                        # Right after opening lead (trick 1, offset 0), non-dummy seats request dummy!
                        if trick_no == 1 and offset == 0 and seat != dummy_seat:
                            send(f"{seat} ready for{sp}dummy")
                            dummy_line = recv()
                            if not dummy_line:
                                disconnected = True
                                break
                            dummy_cards = self._parse_hand_line(dummy_line)

                    if disconnected:
                        break
                    leader_idx = self._winner(trick_cards, last_bid_denom, rank_chars, suit_map)

                if disconnected:
                    break
                if not recv():
                    break

    @staticmethod
    def _parse_hand_line(line: str) -> list[str]:
        body = line.split(":", 1)[1].strip()
        cards: list[str] = []
        for part in body.split("."):
            tokens = part.strip().split()
            if len(tokens) >= 2 and tokens[1] != "-":
                s_char = tokens[0].lower()
                for r_tok in tokens[1:]:
                    cards.append(r_tok.lower() + s_char)
        return cards

    @staticmethod
    def _pick_legal(
        hand: list[str],
        led_suit: str,
        *,
        pick_highest: bool = False,
        ranks: str = "23456789tjqka",
    ) -> str:
        candidates = [c for c in hand if c[1] == led_suit] if led_suit else []
        if not candidates:
            candidates = list(hand)
        if pick_highest:
            return max(candidates, key=lambda c: ranks.index(c[0]))
        return candidates[0]

    @staticmethod
    def _winner(
        trick: list[tuple[int, str]], trump: str, ranks: str, suits: dict[str, int]
    ) -> int:
        led_suit = trick[0][1][1]
        best_p, best_c = trick[0]
        t_char = trump.lower() if trump in suits else ""
        for p_idx, card in trick[1:]:
            is_t = bool(t_char and card[1] == t_char)
            best_t = bool(t_char and best_c[1] == t_char)
            if is_t and not best_t:
                best_p, best_c = p_idx, card
            elif card[1] == best_c[1] and ranks.index(card[0]) > ranks.index(best_c[0]):
                if is_t or card[1] == led_suit:
                    best_p, best_c = p_idx, card
        return best_p


def test_bluechip_package_exports():
    for symbol in (
        "BlueChipLineTransport",
        "BlueChipSeatBot",
        "SocketBlueChipController",
        "SubprocessTcpBlueChipController",
    ):
        assert hasattr(tenace.harness, symbol)
        assert hasattr(tenace, symbol)


def test_bluechip_wire_formatting_helpers():
    assert format_bluechip_deal_header(1, dealer=Seat.NORTH, ns_vul=False, ew_vul=False) == (
        "Board number 1. Dealer North. Neither vulnerable."
    )
    assert format_bluechip_deal_header(2, dealer=Seat.EAST, ns_vul=True, ew_vul=False) == (
        "Board number 2. Dealer East. N/S vulnerable."
    )
    assert format_bluechip_deal_header(3, dealer=Seat.SOUTH, ns_vul=False, ew_vul=True) == (
        "Board number 3. Dealer South. E/W vulnerable."
    )
    assert format_bluechip_deal_header(4, dealer=Seat.WEST, ns_vul=True, ew_vul=True) == (
        "Board number 4. Dealer West. Both vulnerable."
    )
    with pytest.raises(ValueError):
        format_bluechip_deal_header(0, dealer=Seat.NORTH, ns_vul=False, ew_vul=False)
    with pytest.raises(ValueError):
        format_bluechip_deal_header(True, dealer=Seat.NORTH, ns_vul=False, ew_vul=False)

    # Hand formatting uses WBF BlueChip v18 order: S, H, D, C with void marker "-"
    sample_hand = [
        card_id(Suit.CLUBS, 12),  # CA
        card_id(Suit.CLUBS, 11),  # CK
        card_id(Suit.CLUBS, 0),  # C2
        card_id(Suit.HEARTS, 10),  # HQ
        card_id(Suit.HEARTS, 7),  # H9
        card_id(Suit.HEARTS, 5),  # H7
        card_id(Suit.HEARTS, 3),  # H5
        card_id(Suit.HEARTS, 1),  # H3
        card_id(Suit.SPADES, 9),  # SJ
        card_id(Suit.SPADES, 8),  # ST
        card_id(Suit.SPADES, 6),  # S8
        card_id(Suit.SPADES, 4),  # S6
        card_id(Suit.SPADES, 2),  # S4
    ]
    formatted = format_bluechip_hand(sample_hand)
    assert formatted == "S J T 8 6 4. H Q 9 7 5 3. D -. C A K 2."
    # Verify exact PBN output under BEN TMClient.parse_hand() stripping rules
    wire_line = f"North's cards: {formatted}"
    pbn = (
        wire_line[wire_line.index(":") + 1 : wire_line.rindex(".")]
        .replace(" ", "")
        .replace("-", "")
        .replace("S", "")
        .replace("H", "")
        .replace("D", "")
        .replace("C", "")
    )
    assert pbn == "JT864.Q9753..AK2"
    with pytest.raises(ValueError):
        format_bluechip_hand(sample_hand[:12])
    with pytest.raises(ValueError):
        format_bluechip_hand(sample_hand[:12] + [sample_hand[0]])

    # Action conversions
    assert openspiel_action_to_bluechip(52) == "PASSES"
    assert openspiel_action_to_bluechip(53) == "DOUBLES"
    assert openspiel_action_to_bluechip(54) == "REDOUBLES"
    assert openspiel_action_to_bluechip(55) == "bids 1C"
    assert openspiel_action_to_bluechip(59) == "bids 1NT"
    assert openspiel_action_to_bluechip(89) == "bids 7NT"
    assert openspiel_action_to_bluechip(51) == "plays as"
    assert openspiel_action_to_bluechip(44) == "plays kc"
    with pytest.raises(ValueError):
        openspiel_action_to_bluechip(-1)
    with pytest.raises(ValueError):
        openspiel_action_to_bluechip(90)
    with pytest.raises(ValueError):
        openspiel_action_to_bluechip(True)

    # Parsing player action lines including alerts, explanations, card-info dots, and casing
    assert parse_bluechip_player_action("NORTH PASSES") == 52
    assert parse_bluechip_player_action("North passes") == 52
    assert parse_bluechip_player_action("North passes.") == 52
    assert parse_bluechip_player_action("EAST DOUBLES Alert.") == 53
    assert parse_bluechip_player_action("East doubles") == 53
    assert parse_bluechip_player_action("SOUTH REDOUBLES") == 54
    assert parse_bluechip_player_action("South redoubles") == 54
    assert parse_bluechip_player_action("NORTH bids 1NT Alert.") == 59
    assert parse_bluechip_player_action("North bids 1NT. Infos. 15-17") == 59
    assert parse_bluechip_player_action("  WEST   bids   2C   Alert. Stayman  ") == 60
    assert parse_bluechip_player_action("SOUTH plays kc") == 44
    assert parse_bluechip_player_action("SOUTH plays as") == 51
    assert parse_bluechip_player_action("East plays AS. Forced") == 51
    assert parse_bluechip_player_action("East plays KC.") == 44
    with pytest.raises(ValueError):
        parse_bluechip_player_action("NORTH bids 8NT")
    with pytest.raises(ValueError):
        parse_bluechip_player_action("NORTH plays 2Sx")
    with pytest.raises(ValueError):
        parse_bluechip_player_action("INVALID LINE")


def test_bluechip_seat_bot_plays_multi_board_duplicate_match_over_real_tcp():
    peers: list[LoopbackBlueChipPeer] = []

    def make_bluechip_bot(seat: Seat) -> BlueChipSeatBot:
        def controller_factory() -> SocketBlueChipController:
            ctrl = SocketBlueChipController(timeout_secs=5.0)
            peer = LoopbackBlueChipPeer(port=ctrl.port, open_1nt_once=True)
            peers.append(peer)
            peer.start()
            ctrl.accept()
            return ctrl

        return BlueChipSeatBot(seat=seat, controller_factory=controller_factory)

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    summary = runner.play_match(
        team_a_factory=make_bluechip_bot,
        team_b_factory=FirstLegalPassBaselineBot,
        num_boards=2,
    )

    assert len(summary.boards) == 2
    for p in peers:
        p.join(timeout=5.0)
    # Confirm Board 1 used NORTH / Neither and Board 2 used EAST / N/S vulnerable
    # on the SAME persistent connections (4 BlueChip bots total across Open & Closed rooms)!
    assert len(peers) == 4
    for p in peers:
        assert len(p.deal_headers) == 2
    all_headers = [h for p in peers for h in p.deal_headers]
    assert any("Board number 1. Dealer North. Neither vulnerable." == h for h in all_headers)
    assert any("Board number 2. Dealer East. N/S vulnerable." == h for h in all_headers)


def test_bluechip_seat_bot_dummy_lead_and_opening_leader_dummy_reveal():
    """Verify:
    1. BlueChipSeatBot as opening leader reveals dummy right after its own opening lead
       before syncing Trick 1 card 2.
    2. BlueChipSeatBot as declarer sends 'Dummy to lead' when Dummy wins a trick and leads next.
    3. WBridge5 double-spaces ('ready for  NORTH's bid') are tolerated.
    """
    game = pyspiel.load_game(
        "bridge",
        {
            "dealer": int(Seat.NORTH),
            "dealer_vul": False,
            "non_dealer_vul": False,
            "use_double_dummy_result": False,
        },
    )
    state = game.new_initial_state()
    # Construct a deterministic deal where SOUTH (Dummy when NORTH declares 1NT) holds
    # all high cards in Clubs so Dummy is guaranteed to win Trick 1 and lead Trick 2!
    # Deal order in OpenSpiel: action idx % 4 -> 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST.
    north_cards = [card_id(Suit.DIAMONDS, r) for r in range(13)]
    east_cards = [card_id(Suit.CLUBS, r) for r in range(6)] + [
        card_id(Suit.HEARTS, r) for r in range(7)
    ]  # East holds C2..C7 and H2..H8 -> East leads C2 on Trick 1!
    south_cards = [card_id(Suit.CLUBS, r) for r in range(6, 13)] + [
        card_id(Suit.HEARTS, r) for r in range(7, 13)
    ]  # South (Dummy) holds C8..CA and H9..HA -> South wins Trick 1 in Clubs!
    west_cards = [card_id(Suit.SPADES, r) for r in range(13)]

    for idx in range(13):
        state.apply_action(north_cards[idx])
        state.apply_action(east_cards[idx])
        state.apply_action(south_cards[idx])
        state.apply_action(west_cards[idx])

    peers: dict[Seat, LoopbackBlueChipPeer] = {}
    bots: dict[Seat, BlueChipSeatBot] = {}

    def make_bot(seat: Seat, open_1nt: bool) -> BlueChipSeatBot:
        def factory() -> SocketBlueChipController:
            ctrl = SocketBlueChipController(timeout_secs=5.0)
            peer = LoopbackBlueChipPeer(
                port=ctrl.port,
                open_1nt_once=open_1nt,
                wbridge5_double_spaces=True,
                pick_highest_card=True,
            )
            peers[seat] = peer
            peer.start()
            ctrl.accept()
            return ctrl

        return BlueChipSeatBot(seat=seat, controller_factory=factory)

    try:
        # NORTH opens 1NT (declarer), EAST is opening leader (BlueChipSeatBot!),
        # SOUTH is Dummy, WEST is defender.
        bots[Seat.NORTH] = make_bot(Seat.NORTH, open_1nt=True)
        bots[Seat.EAST] = make_bot(Seat.EAST, open_1nt=False)
        bots[Seat.SOUTH] = FirstLegalPassBaselineBot(Seat.SOUTH)  # type: ignore[assignment]
        bots[Seat.WEST] = FirstLegalPassBaselineBot(Seat.WEST)  # type: ignore[assignment]

        for bot in bots.values():
            bot.reset_for_board()

        while not state.is_terminal():
            acting = Seat(state.current_player())
            act = bots[acting].step(state.clone())
            assert act in state.legal_actions()
            state.apply_action(act)

        for bot in bots.values():
            on_end = getattr(bot, "on_board_end", None)
            if callable(on_end):
                on_end(state.clone())
    finally:
        for bot in bots.values():
            close_fn = getattr(bot, "close", None)
            if callable(close_fn):
                close_fn()
        for peer in peers.values():
            peer.join(timeout=5.0)

    # 1. Opening leader EAST sent its opening lead, then received Dummy's cards BEFORE Trick 1 card 2
    east_log = peers[Seat.EAST].event_log
    idx_east_lead = next(i for i, e in enumerate(east_log) if e.startswith("SEND: East plays "))
    idx_east_req_dummy = next(i for i, e in enumerate(east_log) if "ready for  dummy" in e)
    idx_east_recv_dummy = next(
        i for i, e in enumerate(east_log) if e.startswith("RECV: Dummy's cards:")
    )
    idx_east_trick1_dummy_card = next(
        i for i, e in enumerate(east_log) if "Dummy's card to trick 1" in e
    )
    assert idx_east_lead < idx_east_req_dummy < idx_east_recv_dummy < idx_east_trick1_dummy_card

    # 2. Declarer NORTH received "Dummy to lead" after Dummy (SOUTH) won Trick 1!
    assert "Dummy to lead" in peers[Seat.NORTH].to_lead_prompts


def test_subprocess_tcp_bluechip_controller_spawns_and_cleans_up_process():
    for bad_timeout in (0, -1.0, float("nan"), float("inf"), float("-inf"), True, False):
        with pytest.raises(ValueError, match="timeout_secs must be a finite positive number"):
            SocketBlueChipController(timeout_secs=bad_timeout)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"\{port\}"):
        SubprocessTcpBlueChipController(cmd_template=f"{sys.executable} -c 'pass'")

    child_code = (
        "import socket, sys, time\n"
        "suits = {'C': 0, 'D': 1}\n"
        "s = socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=5.0)\n"
        's.sendall(b\'Connecting "SubprocBot" as ANYPL using protocol version 18\\r\\n\')\n'
        "line = s.makefile('r', encoding='ascii', newline='\\r\\n').readline().strip()\n"
        "s.sendall((line + ' ACK\\r\\n').encode('ascii'))\n"
        "time.sleep(30)\n"
    )
    cmd_template = f"{sys.executable} -c {shlex.quote(child_code)} {{port}}"
    ctrl = SubprocessTcpBlueChipController(cmd_template=cmd_template, timeout_secs=5.0)
    assert isinstance(ctrl, BlueChipLineTransport)
    try:
        greeting = ctrl.read_line()
        assert greeting == 'Connecting "SubprocBot" as ANYPL using protocol version 18'
        ctrl.send_line('NORTH ("SubprocBot") seated')
        ack = ctrl.read_line()
        assert ack == 'NORTH ("SubprocBot") seated ACK'
        proc = ctrl._process
        assert proc is not None
        assert proc.poll() is None
    finally:
        ctrl.close()

    assert proc.poll() is not None
    # Idempotent close
    ctrl.close()

    # Optional {seat} placeholder requires a seat argument and expands to Title Case
    with pytest.raises(ValueError, match=r"\{seat\}"):
        SubprocessTcpBlueChipController(
            cmd_template=f"{sys.executable} -c 'pass' {{port}} {{seat}}",
            timeout_secs=1.0,
        )

    seat_child = (
        "import socket, sys, time\n"
        "port = int(sys.argv[1])\n"
        "seat = sys.argv[2]\n"
        "s = socket.create_connection(('127.0.0.1', port), timeout=5.0)\n"
        "msg = f'Connecting \"BEN\" as {seat} using protocol version 18 GIB-BBO 0.9.0.4\\r\\n'\n"
        "s.sendall(msg.encode('ascii'))\n"
        "line = s.makefile('r', encoding='ascii', newline='\\r\\n').readline().strip()\n"
        "s.sendall((seat + '::' + line + '\\r\\n').encode('ascii'))\n"
        "time.sleep(30)\n"
    )
    seat_cmd = f"{sys.executable} -c {shlex.quote(seat_child)} {{port}} {{seat}}"
    seat_ctrl = SubprocessTcpBlueChipController(
        cmd_template=seat_cmd,
        timeout_secs=5.0,
        seat=Seat.EAST,
    )
    try:
        greeting = seat_ctrl.read_line()
        assert greeting.startswith('Connecting "BEN" as East using protocol version 18 ')
        seat_ctrl.send_line('East ("BEN") seated')
        assert seat_ctrl.read_line() == 'East::East ("BEN") seated'
    finally:
        seat_ctrl.close()


def test_bluechip_seat_bot_accepts_greeting_with_trailing_config_info():
    class ScriptedTransport:
        def __init__(self, lines: list[str]) -> None:
            self._lines = list(lines)
            self.sent: list[str] = []

        def read_line(self) -> str:
            return self._lines.pop(0)

        def send_line(self, line: str) -> None:
            self.sent.append(line)

        def close(self) -> None:
            return None

    transport = ScriptedTransport(
        [
            'Connecting "BEN" as North using protocol version 18 GIB 0.9.0.4 config/default.conf',
            "North ready for teams",
            "North ready to start",
            "North ready for deal",
            "North ready for cards",
            "North passes",
        ]
    )
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state()
    for c in range(52):
        state.apply_action(c)

    bot = BlueChipSeatBot(
        seat=Seat.NORTH,
        controller_factory=lambda: transport,
        team_ns="BlueChip",
        team_ew="Tenace",
    )
    assert bot.step(state.clone()) == PASS_ACTION
    assert transport.sent[0] == 'North ("BEN") seated'
    assert transport.sent[1] == 'Teams : N/S : "BlueChip" E/W : "Tenace"'
    assert transport.sent[2] == "Start of board"
    assert transport.sent[3] == "Board number 1. Dealer North. Neither vulnerable."
    assert transport.sent[4].startswith("North's cards: S ")


def test_bluechip_seat_bot_reset_for_board_recovers_if_unfinished_board_aborted():
    peers: list[LoopbackBlueChipPeer] = []

    def controller_factory() -> SocketBlueChipController:
        ctrl = SocketBlueChipController(timeout_secs=5.0)
        peer = LoopbackBlueChipPeer(port=ctrl.port, open_1nt_once=False)
        peers.append(peer)
        peer.start()
        ctrl.accept()
        return ctrl

    bot = BlueChipSeatBot(seat=Seat.NORTH, controller_factory=controller_factory)
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    try:
        state1 = game.new_initial_state()
        for c in range(52):
            state1.apply_action(c)
        assert bot.step(state1.clone()) == PASS_ACTION

        # Abort board 1 mid-auction without calling on_board_end; reset_for_board must
        # cleanly close the desynchronized session and reconnect on board 2.
        bot.reset_for_board()
        assert len(peers) == 1

        state2 = game.new_initial_state()
        for c in range(52):
            state2.apply_action(c)
        assert bot.step(state2.clone()) == PASS_ACTION
        assert len(peers) == 2
    finally:
        bot.close()
        for p in peers:
            p.join(timeout=5.0)


def test_bluechip_seat_bot_rejects_use_double_dummy_result_true():
    peers: list[LoopbackBlueChipPeer] = []

    def make_bluechip_bot(seat: Seat) -> BlueChipSeatBot:
        def controller_factory() -> SocketBlueChipController:
            ctrl = SocketBlueChipController(timeout_secs=5.0)
            peer = LoopbackBlueChipPeer(port=ctrl.port, open_1nt_once=True)
            peers.append(peer)
            peer.start()
            ctrl.accept()
            return ctrl

        return BlueChipSeatBot(seat=seat, controller_factory=controller_factory)

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=True)
    with pytest.raises(ValueError, match="use_double_dummy_result=False"):
        runner.play_match(
            team_a_factory=make_bluechip_bot,
            team_b_factory=FirstLegalPassBaselineBot,
            num_boards=1,
        )
    for p in peers:
        p.join(timeout=5.0)

    # Also reject bare pyspiel.load_game("bridge") whose C++ default is use_double_dummy_result=True
    default_game = pyspiel.load_game("bridge")
    default_state = default_game.new_initial_state()
    for c in range(52):
        default_state.apply_action(c)
    bot = BlueChipSeatBot(
        seat=Seat.NORTH,
        controller_factory=lambda: pytest.fail("should reject before connecting"),
    )
    with pytest.raises(ValueError, match="use_double_dummy_result=False"):
        bot.step(default_state)

    for _ in range(4):
        default_state.apply_action(PASS_ACTION)
    with pytest.raises(ValueError, match="use_double_dummy_result=False"):
        bot.on_board_end(default_state)


def test_bluechip_seat_bot_as_dummy_syncs_all_52_cards_and_plays_board_2():
    """When partner opens 1NT and BlueChipSeatBot sits in dummy_seat, its step() is never
    called during all 13 tricks; on_board_end() must sync all 52 cards without requesting
    Dummy's cards, send Timing, and transition cleanly onto Board 2 on the same TCP session.
    """
    south_peers: list[LoopbackBlueChipPeer] = []

    def make_south_bot() -> BlueChipSeatBot:
        def controller_factory() -> SocketBlueChipController:
            ctrl = SocketBlueChipController(timeout_secs=5.0)
            peer = LoopbackBlueChipPeer(port=ctrl.port, open_1nt_once=False)
            south_peers.append(peer)
            peer.start()
            ctrl.accept()
            return ctrl

        return BlueChipSeatBot(seat=Seat.SOUTH, controller_factory=controller_factory)

    class NorthOpens1NTBot(FirstLegalPassBaselineBot):
        def step(self, state: pyspiel.State) -> int:
            if state.current_phase() == 1 and 59 in state.legal_actions():
                return 59
            return super().step(state)

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    south_bot = make_south_bot()
    seats = {
        Seat.NORTH: NorthOpens1NTBot(Seat.NORTH),
        Seat.SOUTH: south_bot,
        Seat.EAST: FirstLegalPassBaselineBot(Seat.EAST),
        Seat.WEST: FirstLegalPassBaselineBot(Seat.WEST),
    }
    try:
        board1 = runner._new_dealt_board_state(1)
        runner._play_room(board1, seats)
        assert board1.is_terminal()

        board2 = runner._new_dealt_board_state(2)
        runner._play_room(board2, seats)
        assert board2.is_terminal()
    finally:
        south_bot.close()
        for p in south_peers:
            p.join(timeout=5.0)

    assert len(south_peers) == 1
    peer = south_peers[0]
    assert len(peer.deal_headers) == 2
    assert not any("ready for dummy" in ev for ev in peer.event_log)
    play_syncs = [ev for ev in peer.event_log if "RECV:" in ev and " plays " in ev]
    assert len(play_syncs) == 104  # 52 cards on Board 1 + 52 cards on Board 2


def test_bluechip_seat_bot_rejects_mismatched_reply_actor_seat():
    class ScriptedTransport:
        def __init__(self, lines: list[str]) -> None:
            self._lines = list(lines)
            self.sent: list[str] = []

        def read_line(self) -> str:
            return self._lines.pop(0)

        def send_line(self, line: str) -> None:
            self.sent.append(line)

        def close(self) -> None:
            return None

    # Auction: bot is NORTH, engine illegally replies "SOUTH PASSES" on NORTH's turn
    bad_auction_transport = ScriptedTransport(
        [
            'Connecting "MockEngine" as ANYPL using protocol version 18',
            "NORTH ready for teams",
            "NORTH ready to start",
            "NORTH ready for deal",
            "NORTH ready for cards",
            "SOUTH PASSES",
        ]
    )
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state()
    for c in range(52):
        state.apply_action(c)

    bot = BlueChipSeatBot(seat=Seat.NORTH, controller_factory=lambda: bad_auction_transport)
    with pytest.raises(ValueError, match="expected actor NORTH.*got SOUTH"):
        bot.step(state.clone())

    # Play from Dummy: NORTH is declarer in 1NT, EAST leads Trick 1, now NORTH plays SOUTH (Dummy)
    # Engine illegally replies "NORTH plays ..." instead of "SOUTH plays ..."!
    state.apply_action(59)  # NORTH bids 1NT
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    east_lead = state.legal_actions()[0]
    state.apply_action(east_lead)  # EAST leads; current_player() is NORTH playing Dummy (SOUTH)

    bad_dummy_play_transport = ScriptedTransport(
        [
            'Connecting "MockEngine" as ANYPL using protocol version 18',
            "NORTH ready for teams",
            "NORTH ready to start",
            "NORTH ready for deal",
            "NORTH ready for cards",
            "NORTH ready for NORTH's bid",
            "NORTH ready for EAST's bid",
            "NORTH ready for SOUTH's bid",
            "NORTH ready for WEST's bid",
            "NORTH ready for EAST's card to trick 1",
            "NORTH ready for dummy",
            "NORTH plays 2c",
        ]
    )
    decl_bot = BlueChipSeatBot(
        seat=Seat.NORTH, controller_factory=lambda: bad_dummy_play_transport
    )
    with pytest.raises(ValueError, match="expected actor SOUTH.*got NORTH"):
        decl_bot.step(state.clone())


def test_bluechip_adapter_strict_imperfect_information_fair_play():
    import tenace.harness.bluechip as bluechip_mod

    src = inspect.getsource(bluechip_mod)
    assert "[:52]" not in src
    assert "[:NUM_CARDS]" not in src
    assert ".history()" not in src
