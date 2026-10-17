"""Deterministic synthetic graph generator for the HABEAS demo (PLAN.md §6).

Run:
    python -m src.seed.generate --seed 42 --reset-demo-graph

Safety (PLAN.md §6):
- refuses to touch any graph whose name does not start with the hard-coded
  demo-safe prefix ``HABEAS_demo``;
- refuses to modify a non-empty graph without the explicit
  ``--reset-demo-graph`` flag;
- the reset itself (MATCH (n) DETACH DELETE n) is the only destructive
  operation in the project and lives in trusted seed code, never in agent
  tools.

Everything is derived from ``random.Random(seed)`` and a seeded Faker, so the
same seed reproduces identical IDs, amounts, timestamps, and manifest bytes.
The ground-truth manifest is written next to this module
(src/seed/ground_truth.json, gitignored) and never onto the graph.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from faker import Faker

from src.config import DEMO_GRAPH_PREFIX, Settings, get_settings
from src.db import get_graph, run_query
from src.seed import patterns as P
from src.seed import schema
from src.seed.patterns import (
    AlertTruth,
    FAMILY_BENIGN,
    FAMILY_CYCLE,
    FAMILY_CYCLE_RAPID,
    FAMILY_FAN_IN_OUT,
    FAMILY_RAPID_SHARED,
    GroundTruth,
    PatternTruth,
    SIGNAL_CIRCULAR_TRANSFER,
    SIGNAL_FAN_IN_CONCENTRATION,
    SIGNAL_RAPID_IN_OUT,
    SIGNAL_SHARED_DEVICE,
    SIGNAL_SHARED_IDENTIFIER,
    SIGNAL_SINGLE_FORWARDING_DESTINATION,
    SIGNAL_TEMPORAL_CONSISTENCY,
)

MANIFEST_PATH = Path(__file__).resolve().parent / "ground_truth.json"

# Fixed synthetic 90-day window (PLAN.md §6); never derived from wall clock.
WINDOW_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
WINDOW_DAYS = 90
WINDOW_END = WINDOW_START + timedelta(days=WINDOW_DAYS)

ORDINARY_CUSTOMERS = 330  # + 20 planted customers = 350 total
HIGH_VOLUME_ACCOUNTS = 3
BENIGN_FAN_IN_SENDERS = 8
CYCLE_ROUNDS = 3
RAPID_CHAINS = 5

_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


class SeedRefused(Exception):
    """The reset/seed request violates the PLAN.md §6 safety rules."""


def plan_reset(graph_name: str, reset_flag: bool, nodes: int) -> str:
    """Decide the seeding action or raise SeedRefused. Pure, unit-tested."""
    if not graph_name.startswith(DEMO_GRAPH_PREFIX):
        raise SeedRefused(
            f"refusing to seed or reset graph {graph_name!r}: seed and reset "
            f"operations are limited to graphs named with the demo-safe prefix "
            f"{DEMO_GRAPH_PREFIX!r} (PLAN.md §6)"
        )
    if nodes > 0:
        if not reset_flag:
            raise SeedRefused(
                f"graph {graph_name!r} already contains {nodes} nodes; "
                "re-seeding requires the explicit --reset-demo-graph flag"
            )
        return "reset_and_seed"
    return "seed"


class _Ids:
    """Deterministic zero-padded id allocator (TXN gets 6 digits)."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def next(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        width = 6 if prefix == "TXN" else 4
        return f"{prefix}-{n:0{width}d}"


def _ts(dt: datetime) -> str:
    return dt.strftime(_TS_FMT)


@dataclass
class SeedModel:
    customers: list[dict[str, Any]] = field(default_factory=list)
    accounts: list[dict[str, Any]] = field(default_factory=list)
    owns: list[dict[str, Any]] = field(default_factory=list)
    devices: list[dict[str, Any]] = field(default_factory=list)
    identifiers: list[dict[str, Any]] = field(default_factory=list)
    uses_device: list[dict[str, Any]] = field(default_factory=list)
    has_identifier: list[dict[str, Any]] = field(default_factory=list)
    txn_rows: list[dict[str, Any]] = field(default_factory=list)
    transferred_to: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    triggered_by: list[dict[str, Any]] = field(default_factory=list)
    concerns: list[dict[str, Any]] = field(default_factory=list)
    alert_truth: list[AlertTruth] = field(default_factory=list)
    manifest: GroundTruth | None = None

    @property
    def events(self) -> list[P.Event]:
        return [
            (r["src"], r["dst"], r["txn"]["ts"], r["txn"]["amount"])
            for r in self.txn_rows
        ]


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_model(*, seed: int, graph_name: str) -> SeedModel:
    """Pure deterministic model build — no database or wall-clock access."""
    rng = random.Random(seed)
    faker = Faker()
    faker.seed_instance(seed)
    ids = _Ids()
    model = SeedModel()

    def rand_dt(lo_days: int = 0, hi_days: int = WINDOW_DAYS - 1) -> datetime:
        return WINDOW_START + timedelta(
            days=rng.randint(lo_days, hi_days),
            minutes=rng.randint(0, 23 * 60 + 59),
            seconds=rng.randint(0, 59),
        )

    def new_customer() -> dict[str, Any]:
        return {
            "id": ids.next("CUS"),
            "display_name": faker.name(),
            "created_at": _ts(WINDOW_START - timedelta(days=rng.randint(30, 720))),
        }

    def new_account(owner_id: str) -> dict[str, Any]:
        acc = {
            "id": ids.next("ACC"),
            "status": "active",
            "balance": round(rng.uniform(500, 50_000), 2),
            "currency": "USD",
            "created_at": _ts(WINDOW_START - timedelta(days=rng.randint(1, 365))),
        }
        model.accounts.append(acc)
        model.owns.append({"src": owner_id, "dst": acc["id"], "props": {}})
        return acc

    # ---- ordinary population ------------------------------------------------
    accounts_of: dict[str, list[str]] = {}
    for _ in range(ORDINARY_CUSTOMERS):
        cust = new_customer()
        model.customers.append(cust)
        accounts_of[cust["id"]] = []
        for _ in range(2 if rng.random() < 0.15 else 1):
            acc = new_account(cust["id"])
            accounts_of[cust["id"]].append(acc["id"])

    for cust in model.customers:
        for _ in range(rng.randint(1, 2)):
            device = {
                "id": ids.next("DEV"),
                "first_seen": _ts(rand_dt()),
                "last_seen": _ts(rand_dt()),
            }
            model.devices.append(device)
            model.uses_device.append(
                {"src": cust["id"], "dst": device["id"], "props": {}}
            )
        raw_counters: dict[str, int] = {}
        for kind in rng.sample(["email", "phone", "ip"], k=2):
            raw_counters[kind] = raw_counters.get(kind, 0) + 1
            iid = ids.next("IDE")
            n = raw_counters[kind]
            if kind == "email":
                raw = f"user{iid}@synthetic.invalid"
            elif kind == "phone":
                raw = f"+1-555-01{n:02d}"  # 555-01xx: reserved fictional range
            else:
                raw = f"198.51.100.{n % 254 + 1}"  # RFC 5737 documentation range
            ident = {
                "id": iid,
                "kind": kind,
                "value_hash": _hash(raw),
            }
            model.identifiers.append(ident)
            model.has_identifier.append(
                {"src": cust["id"], "dst": ident["id"], "props": {}}
            )

    ordinary_acc_ids = [a["id"] for a in model.accounts]

    # ---- transactions (all seeded through add_txn for projection invariants)
    projection: dict[tuple[str, str], dict[str, Any]] = {}

    def add_txn(
        src: str, dst: str, dt: datetime, cents: int, *, channel: str | None = None
    ) -> dict[str, Any]:
        assert cents > 0 and WINDOW_START <= dt < WINDOW_END
        txn = {
            "id": ids.next("TXN"),
            "amount": round(cents / 100, 2),
            "currency": "USD",
            "ts": _ts(dt),
            "channel": channel or rng.choice(["wire", "ach", "card", "online"]),
            "status": "settled",
        }
        model.txn_rows.append({"txn": txn, "src": src, "dst": dst})
        key = (src, dst)
        agg = projection.setdefault(key, {"count": 0, "cents": 0, "last_ts": txn["ts"]})
        agg["count"] += 1
        agg["cents"] += cents
        agg["last_ts"] = max(agg["last_ts"], txn["ts"])
        return txn

    ordinary_rows_start = len(model.txn_rows)
    for acc in model.accounts:
        for _ in range(rng.randint(2, 5)):
            dst = rng.choice(ordinary_acc_ids)
            while dst == acc["id"]:
                dst = rng.choice(ordinary_acc_ids)
            add_txn(acc["id"], dst, rand_dt(), rng.randint(500, 200_000))
    ordinary_rows = model.txn_rows[ordinary_rows_start:]

    # Benign traps (PLAN.md §6: noise needs tempting false positives).
    tempting: list[str] = []

    hv_accounts = rng.sample(model.accounts, HIGH_VOLUME_ACCOUNTS)
    hv_first_txn: dict[str, dict[str, Any]] = {}
    for acc in hv_accounts:
        for _ in range(rng.randint(45, 55)):
            dst = rng.choice(ordinary_acc_ids)
            while dst == acc["id"]:
                dst = rng.choice(ordinary_acc_ids)
            txn = add_txn(acc["id"], dst, rand_dt(), rng.randint(500, 200_000))
            hv_first_txn.setdefault(acc["id"], txn)
        tempting.append(acc["id"])

    benign_receiver = rng.choice(model.accounts)
    benign_receiver_id = benign_receiver["id"]
    sender_accs = [a for a in ordinary_acc_ids if a != benign_receiver_id]
    benign_senders = rng.sample(sender_accs, BENIGN_FAN_IN_SENDERS)
    benign_incoming: list[dict[str, Any]] = []
    pay_day = rng.randint(25, 28)
    for i, sender in enumerate(benign_senders):
        dt = WINDOW_START + timedelta(
            days=(i % 3) * 30 + pay_day - 1,
            hours=rng.randint(6, 9),
            minutes=rng.randint(0, 59),
        )
        txn = add_txn(
            sender, benign_receiver_id, dt, rng.randint(50_000, 150_000),
            channel="ach",
        )
        benign_incoming.append(txn)
    tempting.append(benign_receiver_id)

    shared_device = rng.choice(model.devices)
    innocent_pair = rng.sample(model.customers, 2)
    for cust in innocent_pair:
        model.uses_device.append(
            {"src": cust["id"], "dst": shared_device["id"], "props": {}}
        )
    tempting.append(shared_device["id"])

    # ---- planted family 1: CYCLE (PLAN.md §6.1) -----------------------------
    cycle_customers = [new_customer() for _ in range(4)]
    model.customers.extend(cycle_customers)
    cycle_accounts = [new_account(c["id"]) for c in cycle_customers]
    ring = [a["id"] for a in cycle_accounts]
    cycle_events: list[P.Event] = []
    cycle_leg_txns: dict[tuple[int, int], dict[str, Any]] = {}
    for r in range(CYCLE_ROUNDS):
        base_dt = WINDOW_START + timedelta(
            days=5 + r * 25 + rng.randint(0, 2),
            hours=rng.randint(8, 10),
            minutes=rng.randint(0, 59),
        )
        base_amount = rng.randint(80_000, 100_000)
        leg_dt = base_dt
        for j in range(4):
            leg_dt = leg_dt + timedelta(minutes=rng.randint(30, 120))
            cents = base_amount + rng.randint(-9_000, 9_000)
            src, dst = ring[j], ring[(j + 1) % 4]
            txn = add_txn(src, dst, leg_dt, cents, channel="wire")
            cycle_events.append((src, dst, txn["ts"], txn["amount"]))
            cycle_leg_txns[(r, j)] = txn

    # ---- planted family 2: FAN-IN / CASH-OUT (PLAN.md §6.2) -----------------
    fan_source_accounts: list[str] = []
    for _ in range(8):
        cust = new_customer()
        model.customers.append(cust)
        fan_source_accounts.append(new_account(cust["id"])["id"])
    inter_accounts = []
    for _ in range(2):
        cust = new_customer()
        model.customers.append(cust)
        inter_accounts.append(new_account(cust["id"])["id"])
    hub_cust = new_customer()
    model.customers.append(hub_cust)
    hub_account = new_account(hub_cust["id"])["id"]

    intake_cents: dict[str, int] = {a: 0 for a in inter_accounts}
    intake_rows: dict[str, list[dict[str, Any]]] = {a: [] for a in inter_accounts}
    source_first_txn: dict[str, dict[str, Any]] = {}
    for s_acc in fan_source_accounts:
        for k in range(rng.randint(3, 5)):
            dst = inter_accounts[k % 2]  # every source reaches both intermediaries
            cents = rng.randint(8_000, 60_000)
            txn = add_txn(s_acc, dst, rand_dt(5, 70), cents)
            intake_cents[dst] += cents
            intake_rows[dst].append(txn)
            source_first_txn.setdefault(s_acc, txn)

    forward_first: dict[str, dict[str, Any]] = {}
    for inter in inter_accounts:
        target = int(intake_cents[inter] * 0.85)
        n_fwd = rng.randint(3, 4)
        shares = [target // n_fwd] * n_fwd
        shares[-1] += target - sum(shares)
        last_intake_day = max(
            datetime.strptime(t["ts"], _TS_FMT) for t in intake_rows[inter]
        ).replace(tzinfo=timezone.utc)
        fwd_dt = last_intake_day + timedelta(days=rng.randint(1, 3), hours=9)
        for cents in shares:
            txn = add_txn(inter, hub_account, fwd_dt, cents, channel="wire")
            fwd_dt = fwd_dt + timedelta(hours=rng.randint(2, 8))
            forward_first.setdefault(inter, txn)

    # ---- planted family 3: RAPID IN/OUT + SHARED INFRA (PLAN.md §6.3) -------
    rapid_customers = [new_customer() for _ in range(2)]
    model.customers.extend(rapid_customers)
    rapid_accounts = [new_account(c["id"]) for c in rapid_customers]
    rapid_acc_ids = [a["id"] for a in rapid_accounts]

    rapid_device = {
        "id": ids.next("DEV"),
        "first_seen": _ts(rand_dt()),
        "last_seen": _ts(rand_dt()),
    }
    model.devices.append(rapid_device)
    rapid_ident = {
        "id": ids.next("IDE"),
        "kind": "ip",
        "value_hash": _hash(f"198.51.100.rapid{seed}"),
    }
    model.identifiers.append(rapid_ident)
    for cust in rapid_customers:
        model.uses_device.append(
            {"src": cust["id"], "dst": rapid_device["id"], "props": {}}
        )
        model.has_identifier.append(
            {"src": cust["id"], "dst": rapid_ident["id"], "props": {}}
        )

    feeder_accounts: list[str] = []
    for _ in range(2):
        cust = new_customer()
        model.customers.append(cust)
        feeder_accounts.append(new_account(cust["id"])["id"])
    exit_cust = new_customer()
    model.customers.append(exit_cust)
    exit_account = new_account(exit_cust["id"])["id"]

    rapid_alert_txns: dict[str, dict[str, Any]] = {}
    for i in range(RAPID_CHAINS):
        feeder = feeder_accounts[0] if i < 3 else feeder_accounts[1]
        base_dt = WINDOW_START + timedelta(
            days=rng.randint(3, 85),
            hours=rng.randint(9, 16),
            minutes=rng.randint(0, 59),
        )
        t0 = base_dt
        t1 = t0 + timedelta(minutes=rng.randint(10, 40))
        t2 = t1 + timedelta(minutes=rng.randint(30, 90))
        c1 = rng.randint(100_000, 500_000)
        c2 = int(c1 * rng.uniform(0.92, 0.98))
        c3 = int(c2 * rng.uniform(0.92, 0.98))
        txn_in = add_txn(feeder, rapid_acc_ids[0], t0, c1, channel="wire")
        txn_mid = add_txn(rapid_acc_ids[0], rapid_acc_ids[1], t1, c2, channel="wire")
        txn_out = add_txn(rapid_acc_ids[1], exit_account, t2, c3, channel="wire")
        if i == 0:
            rapid_alert_txns["r1_to_r2"] = txn_mid
        if i == 1:
            rapid_alert_txns["r2_to_exit"] = txn_out
        if i == 2:
            rapid_alert_txns["feeder_to_r1"] = txn_in

    # ---- alerts (21: 10 planted + 11 benign; exact split in manifest) -------
    def add_alert(
        txn: dict[str, Any],
        account_id: str,
        reason: str,
        family: str,
        label: str,
        members: list[str],
        signals: list[str],
    ) -> None:
        aid = ids.next("ALR")
        created = datetime.strptime(txn["ts"], _TS_FMT) + timedelta(minutes=1)
        model.alerts.append({
            "id": aid,
            "transaction_id": txn["id"],
            "reason": reason,
            "created_at": _ts(created),
            "triage_status": "open",
        })
        model.triggered_by.append({"src": aid, "dst": txn["id"], "props": {}})
        model.concerns.append({"src": aid, "dst": account_id, "props": {}})
        model.alert_truth.append(
            AlertTruth(
                alert_id=aid, family=family, label=label,  # type: ignore[arg-type]
                trigger_transaction_id=txn["id"], account_id=account_id,
                expected_account_members=members, signals=signals,
            )
        )

    alert_truth = model.alert_truth

    cycle_members = ring
    for r, j in ((0, 0), (1, 1), (2, 2)):
        txn = cycle_leg_txns[(r, j)]
        add_alert(
            txn, ring[j], "circular_transfer_pattern", FAMILY_CYCLE, "suspicious",
            cycle_members,
            [SIGNAL_CIRCULAR_TRANSFER, SIGNAL_TEMPORAL_CONSISTENCY],
        )
    inter_ids = inter_accounts
    add_alert(
        forward_first[inter_ids[0]], inter_ids[0],
        "coordinated_fan_in_fan_out", FAMILY_FAN_IN_OUT, "suspicious",
        inter_ids + [hub_account] + fan_source_accounts,
        [SIGNAL_FAN_IN_CONCENTRATION, SIGNAL_SINGLE_FORWARDING_DESTINATION],
    )
    add_alert(
        forward_first[inter_ids[1]], hub_account,
        "coordinated_fan_in_fan_out", FAMILY_FAN_IN_OUT, "suspicious",
        inter_ids + [hub_account] + fan_source_accounts,
        [SIGNAL_FAN_IN_CONCENTRATION, SIGNAL_SINGLE_FORWARDING_DESTINATION],
    )
    add_alert(
        source_first_txn[fan_source_accounts[0]], inter_ids[0],
        "coordinated_fan_in_fan_out", FAMILY_FAN_IN_OUT, "suspicious",
        inter_ids + [hub_account] + fan_source_accounts,
        [SIGNAL_FAN_IN_CONCENTRATION, SIGNAL_SINGLE_FORWARDING_DESTINATION],
    )
    add_alert(
        rapid_alert_txns["r1_to_r2"], rapid_acc_ids[0],
        "rapid_movement_shared_infrastructure", FAMILY_RAPID_SHARED, "suspicious",
        rapid_acc_ids + feeder_accounts + [exit_account],
        [SIGNAL_RAPID_IN_OUT, SIGNAL_SHARED_DEVICE, SIGNAL_SHARED_IDENTIFIER],
    )
    add_alert(
        rapid_alert_txns["r2_to_exit"], rapid_acc_ids[1],
        "rapid_movement_shared_infrastructure", FAMILY_RAPID_SHARED, "suspicious",
        rapid_acc_ids + feeder_accounts + [exit_account],
        [SIGNAL_RAPID_IN_OUT, SIGNAL_SHARED_DEVICE, SIGNAL_SHARED_IDENTIFIER],
    )
    add_alert(
        rapid_alert_txns["feeder_to_r1"], rapid_acc_ids[0],
        "rapid_movement_shared_infrastructure", FAMILY_RAPID_SHARED, "suspicious",
        rapid_acc_ids + feeder_accounts + [exit_account],
        [SIGNAL_RAPID_IN_OUT, SIGNAL_SHARED_DEVICE, SIGNAL_SHARED_IDENTIFIER],
    )

    # benign alerts: 2 fan-in-temptation + 2 high-volume + 1 shared device
    # + 6 ordinary samples = 11
    for txn in benign_incoming[:2]:
        add_alert(
            txn, benign_receiver_id, "fan_in_rule", FAMILY_BENIGN, "benign",
            [benign_receiver_id], [],
        )
    for acc_id in [a["id"] for a in hv_accounts[:2]]:
        add_alert(
            hv_first_txn[acc_id], acc_id, "high_volume_rule", FAMILY_BENIGN,
            "benign", [acc_id], [],
        )
    innocent_cust_acc = accounts_of[innocent_pair[0]["id"]][0]
    innocent_txn_row = next(r for r in ordinary_rows if r["src"] == innocent_cust_acc)
    add_alert(
        innocent_txn_row["txn"], innocent_cust_acc, "shared_device_rule",
        FAMILY_BENIGN, "benign", [innocent_cust_acc], [],
    )
    for row in rng.sample(ordinary_rows, 6):
        add_alert(
            row["txn"], row["src"], "random_sampling_rule", FAMILY_BENIGN,
            "benign", [row["src"]], [],
        )

    # ---- planted family 4: SLOW CYCLE + DISTINCT RAPID PASS-THROUGH ---------
    # Appended last so every earlier id, amount, timestamp and manifest entry
    # stays byte-identical to previous seeds (append-only synthetic-data change;
    # see PLAN.md "Corrective Phase 5 pass").
    #
    # Design (policy v2): the circulation evidence (the four ring legs per
    # round) and the rapid evidence (the feeder -> member -> exit pass-through)
    # are genuinely distinct observations, and the pass-through implicates the
    # alerted ring member that the investigation targets. Ring legs are spaced
    # 150-240 minutes apart, far above the 90-minute rapid threshold, so the
    # ring alone never yields a rapid_in_out match.
    ring_rapid_customers = [new_customer() for _ in range(4)]
    model.customers.extend(ring_rapid_customers)
    ring_rapid_accounts = [new_account(c["id"]) for c in ring_rapid_customers]
    ring_rapid = [a["id"] for a in ring_rapid_accounts]
    ring_rapid_events: list[P.Event] = []
    for r in range(CYCLE_ROUNDS):
        base_dt = WINDOW_START + timedelta(
            days=5 + r * 25 + rng.randint(0, 2),
            hours=rng.randint(8, 10),
            minutes=rng.randint(0, 59),
        )
        base_amount = rng.randint(35_000, 55_000)
        leg_dt = base_dt
        for j in range(4):
            leg_dt = leg_dt + timedelta(minutes=rng.randint(150, 240))
            cents = base_amount + rng.randint(-5_000, 5_000)
            src, dst = ring_rapid[j], ring_rapid[(j + 1) % 4]
            txn = add_txn(src, dst, leg_dt, cents, channel="wire")
            ring_rapid_events.append((src, dst, txn["ts"], txn["amount"]))
    ring_rapid_feeder_cust = new_customer()
    ring_rapid_exit_cust = new_customer()
    model.customers.extend([ring_rapid_feeder_cust, ring_rapid_exit_cust])
    ring_rapid_feeder = new_account(ring_rapid_feeder_cust["id"])["id"]
    ring_rapid_exit = new_account(ring_rapid_exit_cust["id"])["id"]
    fast_dt = WINDOW_START + timedelta(
        days=rng.randint(60, 64),
        hours=rng.randint(9, 16),
        minutes=rng.randint(0, 59),
    )
    fast_in_cents = rng.randint(40_000, 60_000)
    fast_out_cents = int(fast_in_cents * rng.uniform(0.90, 0.98))
    ring_rapid_fast_in = add_txn(
        ring_rapid_feeder, ring_rapid[0], fast_dt, fast_in_cents, channel="wire")
    add_txn(
        ring_rapid[0], ring_rapid_exit,
        fast_dt + timedelta(minutes=rng.randint(25, 45)),
        fast_out_cents, channel="wire",
    )
    add_alert(
        ring_rapid_fast_in, ring_rapid[0],
        "circular_transfer_with_rapid_forwarding", FAMILY_CYCLE_RAPID, "suspicious",
        ring_rapid + [ring_rapid_feeder, ring_rapid_exit],
        [SIGNAL_CIRCULAR_TRANSFER, SIGNAL_TEMPORAL_CONSISTENCY, SIGNAL_RAPID_IN_OUT],
    )

    # ---- derived transfer-topology projection (PLAN.md §5) ------------------
    model.transferred_to = [
        {
            "src": src,
            "dst": dst,
            "props": {
                "count": agg["count"],
                "total_amount": round(agg["cents"] / 100, 2),
                "last_ts": agg["last_ts"],
                "active": True,
            },
        }
        for (src, dst), agg in sorted(projection.items())
    ]

    # ---- manifest ------------------------------------------------------------
    manifest = GroundTruth(
        seed=seed,
        graph_name=graph_name,
        counts={
            "nodes:Customer": len(model.customers),
            "nodes:Account": len(model.accounts),
            "nodes:Transaction": len(model.txn_rows),
            "nodes:Device": len(model.devices),
            "nodes:Identifier": len(model.identifiers),
            "nodes:Alert": len(model.alerts),
            "rels:OWNS": len(model.owns),
            "rels:SENT": len(model.txn_rows),
            "rels:RECEIVED_BY": len(model.txn_rows),
            "rels:USES_DEVICE": len(model.uses_device),
            "rels:HAS_IDENTIFIER": len(model.has_identifier),
            "rels:TRANSFERRED_TO": len(model.transferred_to),
            "rels:TRIGGERED_BY": len(model.alerts),
            "rels:CONCERNS": len(model.alerts),
        },
        alerts=alert_truth,
        patterns=[
            PatternTruth(
                family=FAMILY_CYCLE,
                account_members=ring,
                customer_members=[c["id"] for c in cycle_customers],
                temporally_consistent=P.cycle_temporally_consistent(cycle_events),
                notes="3 circulation rounds; legs strictly ordered within each round",
            ),
            PatternTruth(
                family=FAMILY_FAN_IN_OUT,
                account_members=inter_ids + [hub_account] + fan_source_accounts,
                customer_members=[],
                notes="8 low-activity sources feed 2 intermediaries; ~85% forwarded to one hub",
            ),
            PatternTruth(
                family=FAMILY_RAPID_SHARED,
                account_members=rapid_acc_ids + feeder_accounts + [exit_account],
                customer_members=[c["id"] for c in rapid_customers],
                notes="5 in/out chains under 90 minutes; 2 customers share a device and an IP identifier",
            ),
            PatternTruth(
                family=FAMILY_CYCLE_RAPID,
                account_members=ring_rapid + [ring_rapid_feeder, ring_rapid_exit],
                customer_members=[c["id"] for c in ring_rapid_customers],
                ring_members=ring_rapid,
                temporally_consistent=P.cycle_temporally_consistent(ring_rapid_events),
                notes=(
                    "slow ring (150-240 min legs) plus one distinct rapid "
                    "pass-through on the alerted member; circulation and rapid "
                    "observations share no transaction"
                ),
            ),
        ],
        tempting_false_positives=tempting,
    )
    model.manifest = manifest
    return model


def _verify_counts(actual: dict[str, int], expected: dict[str, int]) -> None:
    mismatches = {
        k: (expected.get(k), actual.get(k))
        for k in expected
        if actual.get(k) != expected.get(k)
    }
    if mismatches:
        raise RuntimeError(f"seeded graph counts do not match the model: {mismatches}")


def _insert_model(graph, model: SeedModel) -> None:
    schema.insert_nodes(graph, "Customer", model.customers)
    schema.insert_nodes(graph, "Account", model.accounts)
    schema.insert_relationships(
        graph, "OWNS", model.owns, src_label="Customer", dst_label="Account"
    )
    schema.insert_nodes(graph, "Device", model.devices)
    schema.insert_nodes(graph, "Identifier", model.identifiers)
    schema.insert_relationships(
        graph, "USES_DEVICE", model.uses_device, src_label="Customer", dst_label="Device"
    )
    schema.insert_relationships(
        graph, "HAS_IDENTIFIER", model.has_identifier,
        src_label="Customer", dst_label="Identifier",
    )
    schema.insert_transactions(graph, model.txn_rows)
    schema.insert_relationships(
        graph, "TRANSFERRED_TO", model.transferred_to,
        src_label="Account", dst_label="Account",
    )
    schema.insert_nodes(graph, "Alert", model.alerts)
    schema.insert_relationships(
        graph, "TRIGGERED_BY", model.triggered_by,
        src_label="Alert", dst_label="Transaction",
    )
    schema.insert_relationships(
        graph, "CONCERNS", model.concerns, src_label="Alert", dst_label="Account"
    )


@dataclass
class SeedRunResult:
    counts: dict[str, int]
    manifest_path: Path
    manifest: GroundTruth


def seed(
    settings: Settings,
    *,
    seed: int = 42,
    reset: bool = False,
    graph=None,
    manifest_path: Path | None = None,
) -> SeedRunResult:
    """Seed the demo graph. Caller must already satisfy plan_reset rules.

    ``reset=True`` clears the graph first (trusted seed code only).
    ``manifest_path`` overrides where the manifest is written; the evaluation
    layer uses it so an alternative-seed run can never overwrite the canonical
    ``src/seed/ground_truth.json`` for seed 42.
    """
    if graph is None:
        graph = get_graph(settings)
    if reset:
        schema.clear_graph(graph)
    schema.ensure_indexes(graph)
    model = build_model(seed=seed, graph_name=settings.graph_name)
    _insert_model(graph, model)
    counts = schema.graph_counts(graph)
    _verify_counts(counts, model.manifest.counts)
    target = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    target.write_text(
        model.manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return SeedRunResult(counts=counts, manifest_path=target,
                         manifest=model.manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.seed.generate",
        description="Seed the HABEAS synthetic demo graph (deterministic).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reset-demo-graph", action="store_true",
        help="explicitly clear the demo graph before seeding (required when "
             "it already contains data; refused for non-demo graph names)",
    )
    args = parser.parse_args(argv)
    settings = get_settings()

    if not settings.graph_name.startswith(DEMO_GRAPH_PREFIX):
        print(
            f"REFUSED: {settings.graph_name!r} does not use the demo-safe "
            f"prefix {DEMO_GRAPH_PREFIX!r}; nothing was written.",
            file=sys.stderr,
        )
        return 2

    graph = get_graph(settings)
    run_query(graph, "RETURN 1 AS bootstrap", read_only=False)  # create key if missing
    try:
        action = plan_reset(settings.graph_name, args.reset_demo_graph,
                            schema.node_count(graph))
    except SeedRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    if action == "reset_and_seed":
        schema.clear_graph(graph)

    result = seed(settings, seed=args.seed, reset=False, graph=graph)
    print(f"Seeded graph {settings.graph_name!r} with seed {args.seed}.")
    for key in sorted(result.counts):
        print(f"  {key}: {result.counts[key]}")
    print(f"Ground-truth manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
