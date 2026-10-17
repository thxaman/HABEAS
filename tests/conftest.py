"""Shared pytest fixtures.

The seeded demo graph is used where the real planted data matters; ``test_net``
is a small, fully controlled isolated graph for precise tool/algorithm cases
(direction, hop caps, active filtering, temporal corroboration).

DB-backed fixtures skip cleanly when FalkorDB is unreachable; the phase gates
are run with the database up and must show real (non-skipped) passes.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.config import DEMO_GRAPH_PREFIX, load_settings
from src.db import get_graph, run_query
from src.seed import schema

# Code-owned label used only by test fixtures (not agent-visible).
TEST_LABEL = "Phase0Test"
TEST_NET_SUFFIX = "_test_net"

# Fixed base so ordering is unambiguous and independent of run time.
_BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _ts(minutes: int) -> str:
    return (_BASE + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


# (src, dst, minutes, amount, active) — active=False models a deactivated route.
EVENTS: list[tuple[str, str, int, float, bool]] = [
    # Corroborated 4-cycle: timestamps chain around the ring (two rounds).
    ("TNC1", "TNC2", 0, 10_000.0, True),
    ("TNC2", "TNC3", 30, 9_800.0, True),
    ("TNC3", "TNC4", 60, 9_600.0, True),
    ("TNC4", "TNC1", 90, 9_400.0, True),
    ("TNC1", "TNC2", 120, 10_100.0, True),
    ("TNC2", "TNC3", 150, 9_900.0, True),
    ("TNC3", "TNC4", 180, 9_700.0, True),
    ("TNC4", "TNC1", 210, 9_500.0, True),
    # Structural-only ring: a duplicated leg means sorted events cannot chain,
    # so this cycle must NOT be reported as temporally corroborated.
    ("TNN1", "TNN2", 300, 5_000.0, True),
    ("TNN1", "TNN2", 400, 4_900.0, True),
    ("TNN2", "TNN3", 350, 4_800.0, True),
    ("TNN3", "TNN1", 450, 4_700.0, True),
    # Coordinated fan-in: 3 sources -> intermediary I1 -> single cash-out hub.
    ("TNS1", "TNI1", 500, 10_000.0, True),
    ("TNS2", "TNI1", 520, 10_000.0, True),
    ("TNS3", "TNI1", 540, 10_000.0, True),
    ("TNI1", "TNHUB", 600, 25_500.0, True),
    # Benign salary receiver: same 3 senders, NO forwarding behaviour.
    ("TNS1", "TNBEN", 700, 5_000.0, True),
    ("TNS2", "TNBEN", 720, 5_000.0, True),
    ("TNS3", "TNBEN", 740, 5_000.0, True),
    # Frozen destination: reachable structurally but must be excluded (status).
    ("TNS1", "TNFZ", 760, 7_000.0, True),
    # Deactivated route: excluded from active-path tracing.
    ("TNS2", "TNX1", 780, 8_000.0, False),
    # Corrected-policy qualifying scenario: a slow ring whose internal legs are
    # far too spaced (150 min) to be rapid matches, plus ONE genuinely distinct
    # rapid pass-through on TNQ1 (TNQF funds TNQ1; TNQ1 forwards onward within
    # 30 min at 95% of the amount). Circulation evidence (the ring legs) and
    # rapid evidence (the pass-through pair) share zero transaction ids.
    ("TNQ1", "TNQ2", 1600, 640.0, True),
    ("TNQ2", "TNQ3", 1750, 620.0, True),
    ("TNQ3", "TNQ4", 1900, 600.0, True),
    ("TNQ4", "TNQ1", 2050, 580.0, True),
    ("TNQ1", "TNQ2", 4600, 660.0, True),
    ("TNQ2", "TNQ3", 4750, 640.0, True),
    ("TNQ3", "TNQ4", 4900, 620.0, True),
    ("TNQ4", "TNQ1", 5050, 600.0, True),
    ("TNQ1", "TNQ2", 7600, 680.0, True),
    ("TNQ2", "TNQ3", 7750, 660.0, True),
    ("TNQ3", "TNQ4", 7900, 640.0, True),
    ("TNQ4", "TNQ1", 8050, 620.0, True),
    ("TNQF", "TNQ1", 9120, 2_000.0, True),
    ("TNQ1", "TNQE", 9150, 1_900.0, True),
]


def _build_net(graph) -> None:
    """(Re)build the isolated test network in its own demo-safe graph."""
    schema.clear_graph(graph)

    accounts = [
        "TNC1", "TNC2", "TNC3", "TNC4",
        "TNN1", "TNN2", "TNN3",
        "TNS1", "TNS2", "TNS3",
        "TNI1", "TNHUB", "TNBEN", "TNX1", "TNFZ",
        # Corrected-policy qualifying scenario: slow ring (TNQ1..TNQ4), the
        # pass-through feeder TNQF, and the onward exit TNQE.
        "TNQ1", "TNQ2", "TNQ3", "TNQ4", "TNQF", "TNQE",
    ]
    statuses = {"TNFZ": "frozen"}

    customers, owns = [], []
    for acc in accounts:
        cid = "TCU_" + acc[2:]
        customers.append({"id": cid, "display_name": f"Synthetic {cid}",
                          "created_at": _ts(0)})
        owns.append({"src": cid, "dst": acc, "props": {}})

    schema.insert_nodes(graph, "Customer", customers)
    schema.insert_nodes(
        graph, "Account",
        [{"id": a, "status": statuses.get(a, "active"), "balance": 1000.0,
          "currency": "USD", "created_at": _ts(0)} for a in accounts],
    )
    schema.insert_relationships(
        graph, "OWNS", owns, src_label="Customer", dst_label="Account"
    )

    # Shared infrastructure: two customers share one device and one IP id.
    schema.insert_nodes(graph, "Device", [
        {"id": "TNDEV", "first_seen": _ts(0), "last_seen": _ts(900)},
        {"id": "TNDEV2", "first_seen": _ts(0), "last_seen": _ts(900)},
    ])
    schema.insert_nodes(graph, "Identifier", [
        {"id": "TNIDE", "kind": "ip", "value_hash": "hash-ip-shared"},
        {"id": "TNIDE2", "kind": "email", "value_hash": "hash-email-solo"},
    ])
    schema.insert_relationships(
        graph, "USES_DEVICE",
        [{"src": "TCU_S1", "dst": "TNDEV", "props": {}},
         {"src": "TCU_S2", "dst": "TNDEV", "props": {}},
         {"src": "TCU_S3", "dst": "TNDEV2", "props": {}}],
        src_label="Customer", dst_label="Device",
    )
    schema.insert_relationships(
        graph, "HAS_IDENTIFIER",
        [{"src": "TCU_S1", "dst": "TNIDE", "props": {}},
         {"src": "TCU_S2", "dst": "TNIDE", "props": {}},
         {"src": "TCU_S3", "dst": "TNIDE2", "props": {}}],
        src_label="Customer", dst_label="Identifier",
    )

    txn_rows = []
    for i, (src, dst, minutes, amount, active) in enumerate(EVENTS):
        txn_rows.append({
            "txn": {
                "id": f"TNT{i:03d}", "amount": amount, "currency": "USD",
                "ts": _ts(minutes), "channel": "wire", "status": "settled",
            },
            "src": src, "dst": dst, "_active": active,
        })
    schema.insert_transactions(
        graph, [{"txn": r["txn"], "src": r["src"], "dst": r["dst"]} for r in txn_rows]
    )

    # Derived projection from the event ledger (active flag from EVENTS).
    agg: dict[tuple[str, str], dict] = {}
    for r in txn_rows:
        slot = agg.setdefault(
            (r["src"], r["dst"]),
            {"count": 0, "total": 0.0, "last_ts": r["txn"]["ts"], "active": True},
        )
        slot["count"] += 1
        slot["total"] += r["txn"]["amount"]
        slot["last_ts"] = max(slot["last_ts"], r["txn"]["ts"])
        slot["active"] = slot["active"] and r["_active"]
    schema.insert_relationships(
        graph, "TRANSFERRED_TO",
        [{"src": s, "dst": d,
          "props": {"count": v["count"], "total_amount": round(v["total"], 2),
                    "last_ts": v["last_ts"], "active": v["active"]}}
         for (s, d), v in sorted(agg.items())],
        src_label="Account", dst_label="Account",
    )
    schema.ensure_indexes(graph)

    # Alerts referencing controlled transactions, so the agent can be driven
    # end-to-end against this graph (Phase 4). Alert ids and reasons only —
    # no ground-truth labels of any kind.
    alerts = [
        {"id": "ALRT-CYCLE", "transaction_id": "TNT000",
         "reason": "circular_transfer_pattern", "created_at": _ts(5),
         "triage_status": "open"},
        {"id": "ALRT-FANIN", "transaction_id": "TNT015",
         "reason": "coordinated_fan_in_fan_out", "created_at": _ts(605),
         "triage_status": "open"},
        {"id": "ALRT-BENIGN", "transaction_id": "TNT016",
         "reason": "fan_in_rule", "created_at": _ts(705),
         "triage_status": "open"},
        {"id": "ALRT-STRUCT", "transaction_id": "TNT008",
         "reason": "circular_transfer_pattern", "created_at": _ts(305),
         "triage_status": "open"},
        {"id": "ALRT-THIN", "transaction_id": "TNT012",
         "reason": "random_sampling_rule", "created_at": _ts(505),
         "triage_status": "open"},
        # Second alert about the same account as ALRT-CYCLE: used to exercise
        # cross-alert case reuse (the earlier case is a lead, not truth).
        {"id": "ALRT-CYCLE2", "transaction_id": "TNT003",
         "reason": "circular_transfer_pattern", "created_at": _ts(95),
         "triage_status": "open"},
        # Corrected-policy qualifying scenario: slow ring + a distinct rapid
        # pass-through on TNQ1 (the trigger is the pass-through inbound leg, so
        # the concerned account is the ring member with the rapid pair).
        {"id": "ALRT-QUAL", "transaction_id": "TNT033",
         "reason": "circular_transfer_with_rapid_forwarding",
         "created_at": _ts(9121), "triage_status": "open"},
    ]
    schema.insert_nodes(graph, "Alert", alerts)
    schema.insert_relationships(
        graph, "TRIGGERED_BY",
        [{"src": a["id"], "dst": a["transaction_id"], "props": {}} for a in alerts],
        src_label="Alert", dst_label="Transaction",
    )
    concerns = {
        "ALRT-CYCLE": "TNC1", "ALRT-FANIN": "TNI1", "ALRT-BENIGN": "TNBEN",
        "ALRT-STRUCT": "TNN1", "ALRT-THIN": "TNS1", "ALRT-CYCLE2": "TNC1",
        "ALRT-QUAL": "TNQ1",
    }
    schema.insert_relationships(
        graph, "CONCERNS",
        [{"src": aid, "dst": acc, "props": {}} for aid, acc in concerns.items()],
        src_label="Alert", dst_label="Account",
    )


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def graph(settings):
    try:
        g = get_graph(settings)
        # Write-mode bootstrap: ro_query refuses a not-yet-created graph, and a
        # harmless `RETURN 1` write auto-creates it.
        run_query(g, "RETURN 1 AS ok", read_only=False)
        run_query(g, "RETURN 1 AS ok", read_only=True)
    except Exception as exc:
        pytest.skip(
            f"FalkorDB not reachable ({type(exc).__name__}: {exc}); "
            "start it with: docker compose up -d"
        )
    return g


@pytest.fixture(scope="session")
def test_net(settings):
    """Isolated controlled graph for tool tests (never the demo graph)."""
    name = settings.graph_name + TEST_NET_SUFFIX
    assert name.startswith(DEMO_GRAPH_PREFIX), "test graph must stay demo-safe"
    try:
        g = get_graph(settings, name=name)
        run_query(g, "RETURN 1 AS ok", read_only=False)
    except Exception as exc:
        pytest.skip(f"FalkorDB not reachable ({type(exc).__name__}: {exc})")
    _build_net(g)
    return g


@pytest.fixture(scope="session")
def seeded_graph(graph, settings):
    """Seed the demo graph once per test session (deterministic, seed 42)."""
    from src.seed import generate

    generate.seed(settings, seed=42, reset=True, graph=graph)
    return graph


@pytest.fixture()
def action_net(test_net):
    """A controlled sandbox graph, rebuilt before AND after destructive tests.

    Freeze tests must never depend on the demo graph's mutable state, and must
    leave the shared test graph as they found it.
    """
    _build_net(test_net)
    yield test_net
    _build_net(test_net)


@pytest.fixture()
def sandbox_settings(settings):
    """Settings with the action tools opted in, pointed at the test sandbox."""
    from src.config import Settings

    return Settings(**{
        **settings.model_dump(),
        "graph_name": settings.graph_name + TEST_NET_SUFFIX,
        "habeas_demo_mode": True,
        "habeas_enable_actions": True,
    })


@pytest.fixture()
def test_nodes(graph):
    """Register node ids a test creates; remove those nodes afterwards.

    This cleanup is trusted test infrastructure, not an agent tool: PLAN.md
    bans DELETE/DETACH DELETE only in agent-facing tools.
    """
    ids: list[str] = []

    def register(node_id: str) -> str:
        ids.append(node_id)
        return node_id

    yield register

    for node_id in ids:
        run_query(
            graph,
            f"MATCH (n:{TEST_LABEL} {{id: $id}}) DETACH DELETE n",
            {"id": node_id},
        )