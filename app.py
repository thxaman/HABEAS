"""HABEAS analyst workstation — Streamlit UI (Phase 8.7).

    streamlit run app.py

HABEAS is graph-powered: FalkorDB holds the accounts, transfer relationships
and alerts, and the investigation agent reaches it only through MCP tools that
run approved, parameterized, bounded query templates. This workspace is honest
about every layer: the header names FalkorDB as the graph engine and shows a
connection state derived from a real approved read (never a constant label),
the architecture strip describes the paths that actually exist in this repo,
and the guided demonstration runs the three real scenarios only when the user
asks.

An investigation runs in the background (FakeLLM always works offline; a real
provider needs credentials already present in the environment). While it runs,
the progress events the workflow really emitted drive the timeline, the
evidence ledger, the hypothesis panel and the graph overlay — so the view
grows with the investigation instead of jumping from button to finished
result.

Everything shown comes from backend result objects or from emitted events.
Nothing is fabricated: there is no placeholder evidence, no invented risk
score, no simulated-live animation, no alert-id branch that manufactures an
outcome, and no evaluation metric shown without its artifact's own run
identification.

Safety invariants (regression-tested in tests/test_ui.py,
tests/test_ui_workspace.py, tests/test_ui_phase86.py, tests/test_ui_phase87.py):

- the target graph is fixed to the disposable demo showcase graph: there is no
  graph-name and no Cypher input anywhere in this app;
- the demo graph is reseeded only by the explicit, confirmed control — never by
  a rerun, a page load, an alert selection, the guided demo, or an
  investigation;
- a stored result is never rendered for a different alert than the selected one;
- the connection check reuses the approved ``alert_inbox`` read tool — no new
  endpoint and no free-form query;
- API keys are never collected or displayed; FakeLLM always works offline;
- dynamic text placed into HTML rendered with unsafe_allow_html is escaped.
"""
from __future__ import annotations

import time

import streamlit as st

from src.ui import backend
from src.ui import graph_view as gv
from src.ui import view_models as vm

st.set_page_config(
    page_title="HABEAS — Graph-powered financial investigations on FalkorDB",
    page_icon="🛰️",
    layout="wide",
)

RECOMMENDED_ALERTS = ("ALR-0001", "ALR-0021", "ALR-0004")
POLL_SECONDS = 0.35
MAX_POLLS = 900  # ~5 minutes of live polling before we stop re-running

BRAND = "Graph-powered financial investigations, built on FalkorDB."

INTRO = (
    "HABEAS investigates suspicious financial-transfer alerts using "
    "connected-data reasoning. Its agent queries account and transaction "
    "relationships through MCP, gathers and compares evidence, evaluates a "
    "deterministic action policy, simulates a potential account freeze, and "
    "independently verifies recorded results."
)

DATA_NOTE = (
    "This environment uses synthetic data only. A freeze here is a simulated "
    "mutation of the disposable demo graph — not an action against a real "
    "financial account."
)

#: The implemented investigation path, one short label each. The trusted audit
#: path (report builder + independent verifier) is described separately below
#: the strip because it reaches FalkorDB directly, not through MCP.
ARCH_STAGES = (
    ("FalkorDB graph", "synthetic accounts, transfers, alerts"),
    ("MCP tools", "approved, bounded read tools over stdio"),
    ("HABEAS investigation agent", "the model proposes; deterministic Python validates"),
    ("Policy v2 gate", "deterministic freeze-eligibility checks"),
    ("Sandbox simulation / action", "read-only counterfactual; sandbox-only freeze"),
    ("Independent evidence verifier", "replays approved queries, recomputes hashes"),
)

ARCH_AUDIT_NOTE = (
    "Trusted audit path, shown separately because it is not part of the "
    "agent's MCP flow: the report builder and the independent verifier "
    "read FalkorDB directly through the internal graph client, using the "
    "same approved catalog and never executing Cypher that came from graph "
    "properties, evidence text or model output."
)

_CSS = """
<style>
  .habeas-brand h1 {
    margin: 0; letter-spacing: .18em; font-weight: 700; color: #e8f1fb;
    font-size: 1.7rem; line-height: 1.1;
  }
  .habeas-brand .tagline { margin: .3rem 0 .25rem 0; color: #7dd3fc;
                           font-size: 1.02rem; font-weight: 600; }
  .habeas-brand p { margin: .35rem 0 0 0; color: #a8bcd6; font-size: .93rem;
                    max-width: 88ch; line-height: 1.45; }
  .habeas-brand .data-note { color: #fbbf24; font-size: .84rem;
                             margin-top: .4rem; }
  .statusline { margin: .45rem 0 0 0; }
  .statusline .meta { color: #7d93b2; font-size: .76rem; margin-top: .15rem; }
  .arch { display: flex; flex-wrap: wrap; align-items: stretch; gap: .3rem;
          margin: .6rem 0 .15rem 0; }
  .arch-node { background: #0e1a2c; border: 1px solid #24395a;
               border-radius: .55rem; padding: .4rem .6rem; flex: 1 1 9.5rem;
               min-width: 9rem; }
  .arch-name { color: #cfe0f4; font-size: .8rem; font-weight: 600;
               line-height: 1.25; }
  .arch-detail { color: #7d93b2; font-size: .7rem; line-height: 1.35;
                 margin-top: .18rem; }
  .arch-arrow { color: #3b5478; align-self: center; font-size: .9rem; }
  div[data-testid="stSidebar"] { background: #0b1424; }
  .panel {
    background: #0e1a2c; border: 1px solid #1c2f4d; border-radius: .7rem;
    padding: .7rem .85rem; margin-bottom: .55rem;
  }
  .panel h4 { margin: 0 0 .35rem 0; color: #cfe0f4; font-size: .74rem;
              letter-spacing: .09em; text-transform: uppercase; }
  .chip { display:inline-block; padding:.1rem .55rem; border-radius:999px;
          font-size:.76rem; font-weight:600; margin:0 .3rem .25rem 0;
          border:1px solid transparent; white-space:nowrap; }
  .chip-positive{background:rgba(20,184,166,.16);color:#2dd4bf;border-color:rgba(20,184,166,.55);}
  .chip-watch{background:rgba(245,158,11,.14);color:#fbbf24;border-color:rgba(245,158,11,.5);}
  .chip-ok{background:rgba(34,197,94,.14);color:#4ade80;border-color:rgba(34,197,94,.5);}
  .chip-abstain{background:rgba(100,116,139,.18);color:#cbd5e1;border-color:rgba(100,116,139,.6);}
  .chip-error{background:rgba(239,68,68,.16);color:#f87171;border-color:rgba(239,68,68,.55);}
  .chip-muted{background:rgba(148,163,184,.1);color:#94a3b8;border-color:rgba(148,163,184,.35);}
  .ev { border-left:2px solid #1f3a5f; padding:.28rem 0 .28rem .6rem;
        margin-bottom:.32rem; }
  .ev .t { color:#e8f1fb; font-size:.86rem; font-weight:600; }
  .ev .d { color:#8fa5c2; font-size:.78rem; }
  .ev-tool{border-left-color:#38bdf8;} .ev-error{border-left-color:#ef4444;}
  .ev-action{border-left-color:#14b8a6;} .ev-hypotheses{border-left-color:#a78bfa;}
  .ev-decision{border-left-color:#fbbf24;} .ev-stage{border-left-color:#334e70;}
  .ev-evidence{border-left-color:#22d3ee;} .ev-policy{border-left-color:#fbbf24;}
  .ev-verification{border-left-color:#4ade80;}
  .vacant { background:#0e1a2c; border:1px dashed #24395a; border-radius:.7rem;
            padding:1.1rem; color:#7d93b2; text-align:center; font-size:.9rem; }
  .flow { display:flex; flex-wrap:wrap; align-items:center; gap:.25rem;
          margin:.6rem 0 .1rem 0; }
  .flow .step { font-size:.78rem; padding:.16rem .55rem; border-radius:.45rem;
                border:1px solid #24395a; color:#8fa5c2; background:#0e1a2c; }
  .flow .done { color:#2dd4bf; border-color:rgba(20,184,166,.5); }
  .flow .current { color:#fbbf24; border-color:rgba(245,158,11,.6); }
  .flow .arrow { color:#3b5478; font-size:.8rem; margin:0 .1rem; }
</style>
"""


def chip(label: str, tone: str) -> str:
    return f'<span class="chip chip-{tone}">{vm.escape_html(label)}</span>'


def _init_state() -> None:
    st.session_state.setdefault("job", None)
    st.session_state.setdefault("run", None)
    st.session_state.setdefault("selected_alert", None)
    st.session_state.setdefault("inbox", None)
    st.session_state.setdefault("neighborhood", None)
    st.session_state.setdefault("selected_node", None)
    st.session_state.setdefault("selected_edge", None)
    st.session_state.setdefault("polls", 0)
    # Connection state derives from a real approved read (or stays "not
    # checked"); it is never seeded with a "connected" claim.
    st.session_state.setdefault("graph_status", None)
    st.session_state.setdefault("pending_run", None)


def _reset_view_state() -> None:
    """Selector callback: clear everything scoped to the previous alert."""
    st.session_state.run = None
    st.session_state.job = None
    st.session_state.neighborhood = None
    st.session_state.selected_node = None
    st.session_state.selected_edge = None
    st.session_state.polls = 0
    st.session_state.pending_run = None


def _load_inbox(settings) -> dict:
    if st.session_state.inbox is None:
        st.session_state.inbox = backend.load_alert_inbox(settings)
    return st.session_state.inbox


def _ensure_neighborhood(settings, alert_row: dict) -> dict | None:
    account_id = alert_row.get("concerned_account_id")
    if not account_id:
        st.session_state.neighborhood = {
            "ok": False, "error": "alert has no concerned account"}
        return st.session_state.neighborhood
    if st.session_state.neighborhood is None:
        st.session_state.neighborhood = backend.load_neighborhood(
            account_id, settings=settings
        )
    return st.session_state.neighborhood


# ------------------------------------------------------- run state plumbing


def _job() -> "backend.InvestigationJob | None":
    """The session's investigation job — only when it really is one.

    Session state can hold a value this app did not write (a test sentinel, a
    value from an older session). Every read goes through here so the UI
    degrades to "no live job" instead of crashing on such a value.
    """
    job = st.session_state.get("job")
    return job if isinstance(job, backend.InvestigationJob) else None


def _failed_run(job, settings) -> dict:
    return {
        "alert_id": job.alert_id, "status": "failed", "error": job.error,
        "errors": [job.error], "completed": False, "tool_calls": 0,
        "tool_budget": settings.tool_budget, "evidence": [], "hypotheses": [],
        "action": None, "report": None, "signal_families": [],
        "primary_entities": [], "termination_reason": "unrecoverable_error",
        "mode_label": st.session_state.get("mode_label", ""),
        "graph_name": settings.graph_name,
    }


def _finalize_job(settings) -> None:
    """Once the background job is done, promote its result to the stored run."""
    job = _job()
    if job is None or not job.done or st.session_state.run is not None:
        return
    st.session_state.run = _failed_run(job, settings) if job.error else job.snapshot
    st.session_state.inbox = None  # the status columns may have changed


def _current_view() -> tuple[dict, bool, list[dict]]:
    """The live (partial) or final view, plus the events and partiality flag."""
    job = _job()
    if job is not None:
        events = job.events()
        if job.running:
            return job.partial_snapshot(), True, events
        if job.snapshot is not None:
            return job.snapshot, False, events
        if st.session_state.run is not None:
            return st.session_state.run, False, events
        return job.partial_snapshot(), False, events
    run = st.session_state.run
    if run is not None:
        return run, False, []
    return {}, False, []


def _run_state(view: dict, partial: bool, job) -> str:
    if partial:
        return "running"
    if job is not None:
        return job.status()
    status = (view or {}).get("status")
    if status in ("failed", "configuration_error"):
        return "failed"
    return "completed" if (view or {}).get("completed") else "incomplete"


# ------------------------------------------------------------------ header


def _render_workflow_strip(events: list[dict], *, alert_selected: bool) -> None:
    steps = vm.workflow_progress(events, alert_selected=alert_selected)
    parts: list[str] = []
    marks = {"done": "✓ ", "current": "● ", "pending": "○ "}
    for index, step in enumerate(steps):
        if index:
            parts.append('<span class="arrow">→</span>')
        parts.append(
            f'<span class="step {step["state"]}">{marks[step["state"]]}'
            f'{vm.escape_html(step["label"])}</span>'
        )
    st.markdown(f'<div class="flow">{"".join(parts)}</div>',
                unsafe_allow_html=True)


def _ensure_graph_status(settings) -> dict:
    """The current connection status: real check once, then the cached record.

    The first render performs one approved bounded read (``alert_inbox`` with
    ``limit=1``) and records it. Later renders — including the 0.35 s live
    polling loop — reuse that record and show its age, so the header never
    re-queries the graph in a loop and never upgrades a stale success into a
    fresh claim.
    """
    status = st.session_state.get("graph_status")
    if status is None:
        status = backend.check_graph_status(settings)
        st.session_state.graph_status = status
    return status


def _render_connection_status(settings, *, key_suffix: str = "header") -> None:
    status = _ensure_graph_status(settings)
    label, tone = vm.connection_chip(status)
    age = vm.connection_age(status)
    chips = chip(label, tone)
    if status.get("state") == "connected" and age:
        chips += chip(f"checked {age}", "muted")
        if vm.connection_is_stale(status):
            chips += chip("stale check — re-check before trusting it", "watch")
    st.markdown(f'<div class="statusline">{chips}</div>',
                unsafe_allow_html=True)

    if status.get("state") == "connected":
        tool = status.get("tool") or "approved read"
        query_ids = ", ".join(str(q) for q in status.get("query_ids") or [])
        meta = (f"Derived from the last successful approved MCP read "
                f"(`{tool}`" + (f": `{query_ids}`" if query_ids else "") + "). "
                "It is a point-in-time check, not a live subscription, and it "
                "is not re-run while an investigation is polling.")
        if vm.connection_is_stale(status):
            meta += (" The check is older than five minutes — the graph may "
                     "have changed since.")
        st.markdown(f'<div class="meta">{meta}</div>',
                    unsafe_allow_html=True)
    elif status.get("state") == "unavailable":
        st.caption(
            "The last approved read failed: "
            f"{status.get('error') or 'unknown error'}. The workspace will "
            "still show load errors rather than pretending to be connected."
        )
    else:
        st.caption("No graph check has been performed in this session yet.")
    if st.button("Re-check FalkorDB connection", key=f"graph-status-{key_suffix}"):
        st.session_state.graph_status = None
        st.rerun()


def _render_architecture() -> None:
    parts: list[str] = []
    for index, (name, detail) in enumerate(ARCH_STAGES):
        if index:
            parts.append('<span class="arch-arrow">→</span>')
        parts.append(
            '<div class="arch-node">'
            f'<div class="arch-name">{vm.escape_html(name)}</div>'
            f'<div class="arch-detail">{vm.escape_html(detail)}</div>'
            "</div>"
        )
    st.markdown(f'<div class="arch">{"".join(parts)}</div>',
                unsafe_allow_html=True)
    st.caption(ARCH_AUDIT_NOTE)
    with st.expander("How HABEAS works — accurate data paths and why the graph "
                     "matters"):
        st.markdown(
            "**Investigation path.** The agent reaches FalkorDB only through "
            "the MCP stdio server (`src/tools/server.py`); every read resolves "
            "an approved, parameterized, bounded template from the query "
            "catalog — hop bounds are validated (1–5) and there is no "
            "free-form Cypher anywhere in the loop. The model proposes "
            "investigations; deterministic Python builds every evidence "
            "envelope, hypothesis assessment and signal family."
        )
        st.markdown(
            "**Policy v2 gate.** Freeze eligibility is computed in "
            "deterministic Python (`src/agent/policy.py`) from the recorded "
            "evidence — the model cannot authorize an action. The only write "
            "in the system is the approved sandbox freeze template in "
            "`src/tools/actions.py`, enabled only with the sandbox opt-in and "
            "only against the disposable demo graph. The simulation is "
            "read-only."
        )
        st.markdown(
            "**Trusted audit path** (the note above the strip): report "
            "building and independent verification (`src/report/`) use the "
            "internal graph client directly with the same approved catalog. "
            "The verifier replays stored, parameterized queries read-only and "
            "recomputes row hashes; it never executes Cypher read from graph "
            "properties."
        )
        st.markdown(
            "**Why FalkorDB.** Relationship traversal *is* the investigation: "
            "`detect_closed_paths_v1` walks directed transfer edges to find "
            "closed paths, `trace_paths_out_v1` follows an account's outbound "
            "transfer path, the bounded `alert_neighborhood` queries place an "
            "account in its 1–2 hop context for this workspace, "
            "`shared_identifiers_v1` groups accounts by shared "
            "identifiers/devices, and `fan_in_senders_v1` / "
            "`fan_out_receivers_v1` find concentrated counterparties. "
            "Deterministic Python then derives the `circulation`, "
            "`rapid_forwarding` and `fan_in_out` signal families from exactly "
            "those returned rows — these are the implemented queries behind "
            "the claims; no broader graph capability is implied."
        )
        st.caption("HABEAS is built on FalkorDB; no partnership or endorsement is implied.")


def _render_intro(settings, *, alert_selected: bool, events: list[dict]) -> None:
    st.markdown(
        '<div class="habeas-brand"><h1>HABEAS</h1>'
        f'<p class="tagline">{vm.escape_html(BRAND)}</p>'
        f"<p>{vm.escape_html(INTRO)}</p>"
        f'<p class="data-note">{vm.escape_html(DATA_NOTE)}</p></div>',
        unsafe_allow_html=True,
    )
    _render_connection_status(settings)
    _render_architecture()
    _render_workflow_strip(events, alert_selected=alert_selected)
    st.caption(
        "The strip above reflects what the real run has actually reached — it "
        "advances when a stage's events arrive, never on a timer."
    )
    with st.expander("Technical details (graph, transport, model, safety)"):
        creds = backend.credentials_configured(settings)
        mode = st.session_state.get("mode_label") or (
            f"real provider ({settings.llm_provider})" if creds
            else "FakeLLM (deterministic)"
        )
        status = vm.connection_chip(st.session_state.get("graph_status"))
        st.markdown(
            chip("graph: " + settings.graph_name, "muted")
            + chip(status[0], status[1])
            + chip("agent graph access: MCP stdio", "muted")
            + chip("LLM mode: " + mode, "positive" if creds else "muted")
            + chip("writes: sandbox only", "watch"),
            unsafe_allow_html=True,
        )
        st.caption(
            "The agent reaches the graph only through MCP tool calls using "
            "approved, parameterized, bounded query templates — never "
            "free-form Cypher. Interpretations, signal families and the "
            "intervention decision are computed in deterministic Python, not "
            "by the model. The only write is a simulated freeze inside the "
            "disposable demo graph. No API key is ever collected here."
        )


def _render_inbox(settings) -> dict | None:
    st.markdown("#### Alert inbox")
    inbox = _load_inbox(settings)
    if not inbox.get("ok"):
        st.error(f"Could not load alerts from the graph: {inbox.get('error')}")
        if st.button("Retry loading alerts"):
            st.session_state.inbox = None
            st.rerun()
        return None

    alerts = inbox["alerts"]
    query = st.text_input("Search alerts (id, reason, account)", value="",
                          placeholder="e.g. ALR-001, rapid, ACC-0390",
                          key="alert-search")
    needle = query.strip().lower()

    def matches(item: dict) -> bool:
        if not needle:
            return True
        haystack = " ".join(str(item.get(k) or "") for k in (
            "alert_id", "reason", "concerned_account_id", "case_verdict",
            "action_status", "trigger_transaction",
        )).lower()
        return needle in haystack

    rows = [r for r in alerts if matches(r)]
    st.caption(
        f"{len(alerts)} alert(s) loaded from the graph via `alert_inbox_v1` · "
        f"{sum(1 for a in alerts if a.get('case_id'))} already investigated · "
        f"showing {len(rows)}"
    )

    options = []
    for item in rows:
        star = "★ " if item["alert_id"] in RECOMMENDED_ALERTS else ""
        outcome = item.get("case_verdict") or "not investigated"
        options.append(
            f"{star}{item['alert_id']} · {item.get('reason') or '—'} · {outcome}"
        )
    if not options:
        st.warning("No alerts match the search.")
        return None

    index = 0
    if st.session_state.selected_alert:
        for i, item in enumerate(rows):
            if item["alert_id"] == st.session_state.selected_alert:
                index = i
                break
    choice = st.radio("Select an alert", options, index=index,
                      key="alert_radio", on_change=_reset_view_state)
    row = rows[options.index(choice)]

    st.markdown(
        (chip("recommended example", "watch")
         if row["alert_id"] in RECOMMENDED_ALERTS else "")
        + chip(row.get("triage_status") or "open", "muted")
        + chip(row.get("reason") or "no reason given", "muted"),
        unsafe_allow_html=True,
    )
    if st.session_state.selected_alert != row["alert_id"]:
        st.session_state.selected_alert = row["alert_id"]
        _reset_view_state()
        st.rerun()
    return row


# --------------------------------------------------------------- guided demo


def _scenario_status_tone(status: str) -> str:
    return {"running": "watch", "completed": "positive",
            "incomplete": "watch", "failed": "error"}.get(status, "muted")


def _render_scenario_outcome(view: dict, *, status: str) -> None:
    """The current session's REAL result for one scenario card."""
    summary = vm.guided_result_summary(view, status=status)
    chips = chip(status, _scenario_status_tone(status))
    if summary["verdict"]:
        label, tone = vm.verdict_chip(summary["verdict"])
        chips += chip(label, tone)
    if summary["action_status"]:
        label, tone = vm.action_chip(summary["action_status"])
        chips += chip(label, tone)
    if summary["allowed"] is True:
        chips += chip("policy permitted", "positive")
    elif summary["allowed"] is False:
        chips += chip("policy declined", "abstain")
    st.markdown(chips, unsafe_allow_html=True)
    if summary["blocking_reasons"]:
        st.caption("Blocking reasons: "
                   + "; ".join(str(r) for r in summary["blocking_reasons"]))
    elif summary["qualifying_families"]:
        st.caption("Qualifying provenance: "
                   + ", ".join(str(f) for f in summary["qualifying_families"]))
    if summary["action_status"] == "applied":
        st.caption("Sandbox action only: a simulated freeze recorded in the "
                   "disposable demo graph — no real account was touched.")
    if status == "failed" and summary.get("error"):
        st.caption(f"Error: {summary['error']}")
    elif status == "incomplete" and summary.get("termination_reason"):
        st.caption(f"Run stopped early ({summary['termination_reason']}).")


def _queue_guided_run(alert_id: str) -> None:
    """Select the scenario's real alert and queue its investigation.

    The run starts only because the user pressed the button on this card — no
    page load, rerun or alert switch can trigger it — and nothing is reseeded.
    The inbox search is cleared so the example's alert is always selectable.
    """
    st.session_state.selected_alert = alert_id
    st.session_state.pop("alert_radio", None)  # let the selector re-index
    st.session_state.pop("alert-search", None)  # show the full inbox
    _reset_view_state()
    st.session_state.pending_run = {"alert_id": alert_id}
    st.rerun()


def _render_guided_demo(view: dict, *, status: str) -> None:
    with st.expander("Guided demonstration — three real examples",
                     expanded=True):
        st.caption(
            "Each example loads its real alert and starts the real workflow "
            "with FakeLLM (deterministic, offline). Nothing runs until you "
            "press a button; the page never auto-runs or reseeds the graph, "
            "and outcomes shown are this session's actual results."
        )
        scenarios = backend.list_scenarios()
        for scenario in scenarios:
            st.markdown(f"**{scenario['key']} · {scenario['title']}**")
            st.caption(scenario["stake"])
            st.caption(f"alert `{scenario['alert_id']}`")
            if st.button(f"Run example {scenario['key']}",
                         key=f"guided-run-{scenario['key']}",
                         width="stretch", disabled=status == "running"):
                _queue_guided_run(scenario["alert_id"])
            current = view or {}
            if current.get("alert_id") == scenario["alert_id"]:
                _render_scenario_outcome(current, status=status)
            else:
                st.caption("Not run in this session yet.")


def _render_timeline(events: list[dict], *, status: str, elapsed) -> None:
    st.markdown("#### Activity timeline")
    if not events:
        st.markdown(
            '<div class="vacant">No events yet — press <b>Investigate</b> to run '
            "the real workflow; steps appear here as they happen.</div>",
            unsafe_allow_html=True,
        )
        return
    tone = {"running": "watch", "completed": "positive",
            "incomplete": "watch", "failed": "error"}.get(status, "muted")
    label = {"running": "running", "completed": "completed",
             "incomplete": "incomplete", "failed": "failed"}.get(status, status)
    st.markdown(
        chip(f"{label} · {len(events)} event(s)", tone)
        + (chip(f"{elapsed}s elapsed", "muted") if elapsed is not None else ""),
        unsafe_allow_html=True,
    )
    for event in reversed(events[-40:]):
        kind = event.get("type", "stage")
        ok = event.get("ok")
        mark = "" if ok is None else ("✓ " if ok else "✗ ")
        st.markdown(
            f'<div class="ev ev-{vm.escape_html(kind)}">'
            f'<div class="t">{mark}{vm.escape_html(event.get("title"))}</div>'
            f'<div class="d">{vm.escape_html(event.get("detail") or "")}</div>'
            "</div>",
            unsafe_allow_html=True,
        )


def _render_hypotheses(view: dict, *, partial: bool) -> None:
    st.markdown("#### Competing hypotheses")
    hypotheses = view.get("hypotheses") or []
    if not hypotheses:
        st.caption("Assessed once the investigation has gathered enough evidence.")
        return
    if partial:
        st.caption("Live assessment — re-evaluated as each evidence record lands.")
    for hypothesis in hypotheses:
        label, tone = vm.hypothesis_chip(hypothesis.get("status"))
        st.markdown(
            chip(label, tone) + chip(hypothesis.get("name") or "?", "muted")
            + '<div class="ev"><div class="d">supporting '
            f"{len(hypothesis.get('supporting_evidence') or [])} · contradicting "
            f"{len(hypothesis.get('contradicting_evidence') or [])} · open "
            f"{len(hypothesis.get('uncertainties') or [])}</div></div>",
            unsafe_allow_html=True,
        )


def _render_metric_cards(view: dict, events: list[dict], settings, *,
                         partial: bool) -> None:
    if partial:
        calls = view.get("tool_calls", 0)
        budget = settings.tool_budget
        evidence_count = view.get("evidence_count", 0)
        families = view.get("signal_families") or []
    else:
        summary = vm.summarize_run(view)
        calls = summary.get("tool_calls", 0)
        budget = summary.get("tool_budget") or settings.tool_budget
        evidence_count = summary.get("evidence_count", 0)
        families = summary.get("signal_families") or []
    provenance = (view.get("policy") or {}).get("distinct_provenance_families") or (
        (view.get("action") or {}).get("distinct_provenance_families") or [])
    cards = [
        ("MCP tool calls", f"{calls}/{budget}" if budget else str(calls)),
        ("Evidence records", str(evidence_count)),
        ("Signal families", ", ".join(families) or "—"),
        ("Distinct provenance", ", ".join(provenance) or "—"),
        ("Timeline events", str(len(events))),
    ]
    for col, (label, value) in zip(st.columns(5), cards):
        col.markdown(
            f'<div class="panel"><h4>{vm.escape_html(label)}</h4>'
            '<div style="color:#e8f1fb;font-size:1.15rem;font-weight:700">'
            f"{vm.escape_html(value)}</div></div>",
            unsafe_allow_html=True,
        )


def _render_graph_panel(settings, row: dict, view: dict, *,
                        partial: bool) -> None:
    st.markdown("#### Evidence graph")
    neighborhood = _ensure_neighborhood(settings, row)
    if not neighborhood or not neighborhood.get("ok"):
        st.warning(
            "No neighbourhood data is available for this alert "
            f"({(neighborhood or {}).get('error', 'unknown error')}). Nothing is "
            "drawn rather than inventing nodes."
        )
        return

    evidence_entries = view.get("evidence") or []
    ev_graph = vm.graph_from_evidence(evidence_entries, view.get("primary_entities"))
    graph = vm.merge_graphs(neighborhood, ev_graph,
                            focal=row.get("concerned_account_id"))
    if not graph["accounts"]:
        st.info("The graph returned no accounts for this neighbourhood.")
        return

    # Evidence -> graph highlighting: the selector maps a real evidence record
    # to the accounts/legs THAT record reports. A record with no usable
    # mapping highlights nothing, and the UI says so.
    evidence_options = ["— none —"] + [
        str(entry.get("evidence_id"))
        for entry in evidence_entries if entry.get("evidence_id")
    ]
    focus_label = st.selectbox(
        "Highlight one evidence record in the graph", evidence_options,
        key=f"ev-focus-{row.get('alert_id') or 'alert'}",
        help="Highlights only accounts and transfer legs the selected record "
             "itself reported — never invented links.",
    )
    focus_nodes: list[str] = []
    focus_edges: list[list[str]] = []
    if focus_label != "— none —":
        entry = next((e for e in evidence_entries
                      if str(e.get("evidence_id")) == focus_label), None)
        mapping = vm.evidence_graph_elements(entry)
        graph_accounts = set(graph["accounts"])
        graph_edges = {(e.get("src"), e.get("dst")) for e in graph["edges"]}
        focus_nodes = [n for n in mapping["nodes"] if n in graph_accounts]
        focus_edges = [list(pair) for pair in mapping["edges"]
                       if (pair[0], pair[1]) in graph_edges]
        if focus_nodes or focus_edges:
            st.caption(
                f"Highlighting {len(focus_nodes)} account(s) and "
                f"{len(focus_edges)} recorded leg(s) from `{focus_label}` "
                "(violet overlay). What is not in the record is not shown."
            )
        else:
            st.caption(
                f"`{focus_label}` does not map to accounts or legs drawn here "
                "(it may be policy/metadata evidence, or its entities are "
                "outside this neighbourhood) — nothing is highlighted rather "
                "than inventing a relationship."
            )

    col_graph, col_fit = st.columns([5, 1])
    with col_fit:
        if st.button("Reset view", width="stretch"):
            st.session_state.selected_node = None
            st.session_state.selected_edge = None
            st.rerun()
    figure = gv.build_figure(
        accounts=graph["accounts"],
        edges=graph["edges"],
        focal_account_id=graph["focal"],
        primary_entities=view.get("primary_entities"),
        cycle_edges=graph["cycle_edges"],
        transactions=neighborhood.get("transactions"),
        # Nodes/edges the workflow really reported so far — highlighted live.
        live_nodes=view.get("nodes") if partial else None,
        live_edges=view.get("edges") if partial else None,
        focus_nodes=focus_nodes or None,
        focus_edges=focus_edges or None,
    )
    with col_graph:
        event = st.plotly_chart(
            figure, width="stretch", key="habeas_graph",
            on_select="rerun", selection_mode=("points",),
        )
    selected = gv.selected_from_event(event)

    for note in graph["notes"]:
        st.caption(note)
    legend = (
        "Solid cyan = active routing · dashed amber = deactivated routing · "
        "solid teal = corroborated circulation · dotted slate = structural "
        "connectivity only (not circulation)."
    )
    if partial:
        legend += (" Amber dashed overlay = nodes/legs implicated by evidence "
                   "reported so far in this run.")
    if focus_nodes or focus_edges:
        legend += (" Violet overlay = the accounts/legs of the highlighted "
                   "evidence record.")
    st.caption(legend)

    node_id = st.session_state.selected_node
    edge_key = st.session_state.selected_edge
    if selected and selected.get("kind") == "node":
        node_id, edge_key = selected["account_id"], None
    elif selected and selected.get("kind") == "edge":
        edge_key, node_id = (selected["src"], selected["dst"]), None

    if node_id:
        details = gv.node_details(
            node_id, edges=graph["edges"],
            transactions=neighborhood.get("transactions"),
            cases=[row] if row.get("case_id") else [],
        )
        st.markdown(
            f"#### Node inspector — `{vm.escape_html(node_id)}` "
            + chip(f"out {details['out_degree']}", "muted")
            + chip(f"in {details['in_degree']}", "muted"),
            unsafe_allow_html=True,
        )
        st.caption("Real relationships from the neighbourhood query (aggregates) "
                   "plus any transaction events it returned.")
        linked = vm.evidence_for_entity(evidence_entries, node_id)
        if linked:
            st.markdown("**Evidence mentioning this account**")
            st.markdown(" ".join(chip(eid, "muted") for eid in linked),
                        unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Outgoing (aggregate)**")
            st.dataframe(details["outgoing"], width="stretch", height=180)
        with cols[1]:
            st.markdown("**Incoming (aggregate)**")
            st.dataframe(details["incoming"], width="stretch", height=180)
        if details["transactions"]:
            st.markdown("**Transaction events**")
            st.dataframe(details["transactions"][:40], width="stretch",
                         height=180)
        else:
            st.caption("No transaction events were returned for this account.")
    elif edge_key:
        details = gv.edge_details(
            edge_key[0], edge_key[1], edges=graph["edges"],
            transactions=neighborhood.get("transactions"),
        )
        st.markdown(
            f"#### Edge inspector — `{edge_key[0]} → "
            f"{edge_key[1]}`"
        )
        linked = vm.evidence_for_edge(evidence_entries, edge_key[0], edge_key[1])
        if linked:
            st.markdown("**Evidence covering this leg**")
            st.markdown(" ".join(chip(eid, "muted") for eid in linked),
                        unsafe_allow_html=True)
        if details["edge"]:
            st.markdown("**Aggregate transfer edge (TRANSFERRED_TO)**")
            st.dataframe([details["edge"]], width="stretch")
        else:
            st.caption("This edge comes from investigation evidence, not the "
                       "neighbourhood aggregate query.")
        if details["transactions"]:
            st.markdown("**Underlying transaction events**")
            st.dataframe(details["transactions"], width="stretch")
            if details["timestamps"]:
                st.caption(
                    f"{len(details['transactions'])} event(s) returned · first "
                    f"{details['first_ts']} · last {details['last_ts']}"
                )
            if details["txn_ids"]:
                shown = ", ".join(details["txn_ids"][:8])
                more = len(details["txn_ids"]) - 8
                st.caption(f"transaction ids: {shown}"
                           + (f" (+{more} more)" if more > 0 else ""))
        else:
            st.caption("No individual transactions were returned for this pair "
                       "within the window — only the aggregate edge is known.")
    else:
        st.caption("Click a node or an edge in the graph to inspect it.")


def _render_evidence_tab(view: dict, *, partial: bool) -> None:
    if not view:
        st.markdown('<div class="vacant">Run the investigation to populate the '
                    "evidence ledger.</div>", unsafe_allow_html=True)
        return
    rows = vm.evidence_inspector_rows(
        view.get("evidence") or [], (view.get("report") or {}).get("claims")
    )
    if not rows:
        st.markdown('<div class="vacant">No evidence records yet.</div>',
                    unsafe_allow_html=True)
        return
    if partial:
        st.caption("Live ledger — rows appear as evidence events arrive.")
    st.caption(
        f"{len(rows)} evidence record(s); multi-query records keep every source "
        "query, never just the first."
    )
    table = [
        {
            "evidence": r["evidence_id"], "tool": r["tool"], "stance": r["stance"],
            "queries": r["query_count"], "rows": r["total_rows"],
            "truncated": r["truncated_any"], "checked": r["checked"],
            "verification": r["verification_status"],
        }
        for r in rows
    ]
    st.dataframe(table, width="stretch", height=260)
    alert_id = view.get("alert_id") or "alert"
    st.download_button(
        "Download evidence CSV",
        data=vm.evidence_csv(rows),
        file_name=f"{alert_id}-evidence.csv",
        mime="text/csv",
        key=f"csv-{alert_id}-{len(rows)}",
    )
    st.caption(
        "`checked` means a verification assessment was recorded or attempted — "
        "it does not mean the evidence proved true. The outcome is in "
        "`verification`."
    )
    entries_by_id = {
        str(e.get("evidence_id")): e for e in (view.get("evidence") or [])
        if e.get("evidence_id")
    }
    for r in rows:
        s_label, s_tone = vm.stance_chip(r["stance"])
        v_label, v_tone = vm.verification_chip(r["verification_status"])
        with st.expander(
            f"{r['evidence_id']} · {r['tool']}"
            + ("  · MULTI-QUERY" if r["multi_query"] else "")
            + ("" if r["checked"] else "  · not checked")
        ):
            st.markdown(
                chip(s_label, s_tone) + chip(v_label, v_tone)
                + ("**truncated**" if r["truncated_any"] else ""),
                unsafe_allow_html=True,
            )
            st.caption("queries: " + (", ".join(r["query_ids"]) or "—"))
            mapping = vm.evidence_graph_elements(
                entries_by_id.get(str(r["evidence_id"])))
            if mapping["mapped"]:
                st.caption(
                    f"graph mapping: {len(mapping['nodes'])} account/entity "
                    f"id(s), {len(mapping['edges'])} recorded leg(s)"
                )
            else:
                st.caption(
                    "graph mapping: none — this record does not identify "
                    "accounts or legs, so nothing is drawn for it."
                )
            if r["verification_details"]:
                st.caption(f"verification: {r['verification_details']}")
            if r["mismatched_source_queries"]:
                st.caption("mismatched source queries: "
                           + ", ".join(r["mismatched_source_queries"]))
            if r["interpretation"]:
                st.markdown(f"> {r['interpretation']}")


def _policy_fields(view: dict) -> dict:
    """Policy details from the live policy event or from the final action."""
    policy = dict(view.get("policy") or {})
    if policy:
        return policy
    action = view.get("action") or {}
    return {
        key: action.get(key) for key in (
            "policy_version", "allowed", "checks", "blocking_reasons",
            "distinct_provenance_families", "qualifying_evidence_ids",
            "target_relevant_families",
        )
    }


def _render_decision(view: dict, *, partial: bool) -> None:
    if not view:
        st.markdown('<div class="vacant">The policy decision appears after the '
                    "investigation runs.</div>", unsafe_allow_html=True)
        return
    verdict_label, verdict_tone = vm.verdict_chip(view.get("canonical_verdict"))
    meta_chips = [chip(verdict_label, verdict_tone)]
    if view.get("termination_reason"):
        meta_chips.append(chip(f"termination: {view['termination_reason']}", "muted"))
    if view.get("elapsed_s") is not None:
        meta_chips.append(chip(f"elapsed {view['elapsed_s']}s", "muted"))
    if view.get("errors"):
        meta_chips.append(chip(f"{len(view['errors'])} error(s)", "error"))
    st.markdown("".join(meta_chips), unsafe_allow_html=True)
    truncated = view.get("truncated_evidence") or []
    if truncated:
        shown = ", ".join(str(t) for t in truncated[:6])
        more = len(truncated) - 6
        st.warning(
            f"{len(truncated)} evidence record(s) were truncated by the query "
            f"row limits: {shown}" + (f" (+{more} more)" if more > 0 else "")
            + " — findings based on them are bounded by what was returned."
        )

    policy = _policy_fields(view)
    action = view.get("action") or {}
    if not policy.get("checks") and not action.get("action_status"):
        st.caption("The policy decision appears once the workflow reaches it.")
        return

    a_label, a_tone = vm.action_chip(action.get("action_status"))
    st.markdown(
        "#### Policy v" + vm.escape_html(str(policy.get("policy_version") or "?"))
        + " decision<br>" + chip(a_label, a_tone)
        + chip("permitted" if policy.get("allowed") else "not permitted",
               "positive" if policy.get("allowed") else "abstain"),
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Target-relevant families**")
    c1.write(", ".join(policy.get("target_relevant_families") or []) or "none")
    c2.markdown("**Distinct-provenance families counted**")
    c2.write(", ".join(policy.get("distinct_provenance_families") or []) or "none")
    c3.markdown("**Qualifying evidence**")
    c3.write(", ".join(policy.get("qualifying_evidence_ids") or []) or "none")

    checks = policy.get("checks") or {}
    if checks:
        passed = sum(1 for ok in checks.values() if ok)
        with st.expander(f"Policy checks — {passed}/{len(checks)} passed"):
            for name, ok in sorted(checks.items()):
                st.markdown(("✓ " if ok else "✗ ") + f"`{name}`")
    if policy.get("blocking_reasons"):
        st.markdown("**Blocking reasons**")
        for reason in policy["blocking_reasons"]:
            st.markdown(f"- {reason}")


def _render_audit(view: dict, *, partial: bool) -> None:
    if not view:
        st.markdown('<div class="vacant">Simulation, action outcome and '
                    "verification appear here after the run.</div>",
                    unsafe_allow_html=True)
        return
    action = view.get("action") or {}

    sim = view.get("simulation")
    if sim:
        st.markdown("#### Counterfactual simulation (read-only)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Direct outgoing volume", str(sim.get("direct_outgoing_volume")))
        c2.metric("Downstream before", str(sim.get("downstream_accounts_before")))
        c3.metric("Newly unreachable", str(sim.get("newly_unreachable_count")))
        c4.metric("Active edges disabled", str(sim.get("active_edges_disabled")))
        st.caption(
            (sim.get("metric_semantics") or {}).get("direct_outgoing_volume")
            or "Historical transfer volume exposed to disruption — NOT money "
            "that would be frozen."
        )

    st.markdown("#### Action outcome")
    if action.get("action_status") == "applied":
        st.success(
            "Sandbox freeze applied — action "
            f"`{vm.escape_html(action.get('action_id'))}`; historical evidence "
            "retained, routing edges deactivated."
        )
    elif action.get("action_status") == "declined":
        st.warning("No mutation was applied — the policy declined this action.")
    elif action:
        st.caption(
            f"Action status: `{action.get('action_status')}`."
        )

    post = view.get("post_action") or view.get("verification")
    if post:
        if post.get("account_id"):
            st.caption(
                f"Account `{post.get('account_id')}` current "
                f"status: `{post.get('account_status')}`"
            )
        for name, ok in sorted((post.get("checks") or {}).items()):
            st.markdown(("✓ " if ok else "✗ ") + f"`{name}`")

    report = view.get("report") or {}
    if report.get("error"):
        st.error(f"Report generation failed: {report['error']}")
        return
    if not report:
        if partial:
            st.caption("The generated report and its independent verification "
                       "appear when the run completes.")
        return
    counts = vm.verification_counts(report.get("summary"))
    st.markdown("#### Independent verification (claim level)")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Reproduced / verified", counts["reproduced"])
    v2.metric("Changed after action", counts["changed_after_action"])
    v3.metric("Mismatch", counts["mismatch"])
    v4.metric("Unverifiable", counts["unverifiable"])
    if counts["mismatch"]:
        st.error("Verification found MISMATCHES — see the report.")
    elif counts["unverifiable"]:
        st.caption(f"{counts['unverifiable']} record(s) could not be independently "
                   "checked; they are reported, not hidden.")
    for note in (report.get("notes") or [])[:3]:
        st.caption(note)

    if report.get("markdown"):
        st.download_button(
            "Download report (Markdown)", data=report["markdown"],
            file_name=f"{report.get('case_id')}.md", mime="text/markdown",
            key=f"dl-{view.get('alert_id')}",
        )
        st.caption(f"working copy at `{report.get('written_to')}`")
        with st.expander("View full report"):
            st.markdown(report["markdown"])


# ------------------------------------------------- technical / evaluation


def _eval_artifact() -> dict:
    """Cached loader for the offline evaluation artifact (reload on demand)."""
    if "eval_artifact" not in st.session_state:
        st.session_state.eval_artifact = backend.load_eval_summary()
    return st.session_state.eval_artifact


def _render_technical(settings, view: dict, events: list[dict], *,
                      partial: bool, status: str, graph_status: dict) -> None:
    st.caption(
        "Optional diagnostics. Every value comes from this session's real run "
        "objects, the connection check, or a named offline artifact — nothing "
        "is recomputed for display and no metric is shown without its source."
    )
    st.markdown("#### Graph connection")
    label, tone = vm.connection_chip(graph_status)
    age = vm.connection_age(graph_status)
    st.markdown(
        chip(label, tone) + (chip(f"checked {age}", "muted") if age else ""),
        unsafe_allow_html=True,
    )
    source = graph_status.get("tool") or "—"
    query_ids = ", ".join(str(q) for q in graph_status.get("query_ids") or [])
    st.caption(
        f"state: `{graph_status.get('state') or 'not_checked'}` · source: "
        f"`{source}` · query: `{query_ids or '—'}`"
    )
    if graph_status.get("error"):
        st.caption(f"error: {graph_status.get('error')}")

    st.markdown("#### Current run")
    if not view or not view.get("alert_id"):
        st.caption("No investigation run in this session yet.")
    else:
        if partial:
            calls = view.get("tool_calls", 0)
            budget = settings.tool_budget
            evidence_count = view.get("evidence_count", 0)
        else:
            summary = vm.summarize_run(view)
            calls = summary.get("tool_calls", 0)
            budget = summary.get("tool_budget") or settings.tool_budget
            evidence_count = summary.get("evidence_count", 0)
        st.markdown(
            chip(f"run: {status}", _scenario_status_tone(status))
            + chip(f"tool calls {calls}/{budget}", "muted")
            + chip(f"evidence records {evidence_count}", "muted")
            + chip(f"timeline events {len(events)}", "muted"),
            unsafe_allow_html=True,
        )
        policy = _policy_fields(view)
        if policy.get("checks") or policy.get("allowed") is not None:
            st.markdown(
                chip("policy permitted" if policy.get("allowed")
                     else "policy not permitted",
                     "positive" if policy.get("allowed") else "abstain")
                + chip(f"policy v{policy.get('policy_version') or '?'}", "muted"),
                unsafe_allow_html=True,
            )
            for reason in policy.get("blocking_reasons") or []:
                st.caption(f"blocking reason: {reason}")

        report = view.get("report") or {}
        if report.get("error"):
            st.error(f"Report generation failed: {report['error']}")
        elif report:
            rows = vm.verification_status_rows(report.get("summary"))
            total = sum(row["count"] for row in rows)
            st.markdown(
                chip("report generated", "positive")
                + chip(f"verification: {total} claim(s) checked", "muted"),
                unsafe_allow_html=True,
            )
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True,
                             height=180)
            st.caption(
                f"case `{report.get('case_id')}` · report version "
                f"{report.get('report_version')} · working copy "
                f"`{report.get('written_to')}`"
            )
            for note in (report.get("notes") or [])[:3]:
                st.caption(note)
        else:
            st.caption(
                "No report exists for this run yet — report generation and "
                "independent verification run at the end of the workflow."
            )
        sequence = view.get("tool_sequence") or []
        if sequence:
            with st.expander(f"Tool call sequence — {len(sequence)} call(s)"):
                st.write(", ".join(str(item) for item in sequence))

    st.markdown("#### Evaluation artifact (offline)")
    if st.button("Reload artifact", key="reload-eval-artifact"):
        st.session_state.pop("eval_artifact", None)
        st.rerun()
    artifact = _eval_artifact()
    if not artifact.get("ok"):
        st.caption(artifact.get("reason") or "No evaluation artifact.")
    elif not artifact.get("identified"):
        st.warning(artifact.get("reason")
                   or "Unidentified evaluation artifact; metrics withheld.")
    else:
        dataset = artifact.get("dataset") or {}
        st.markdown(
            "**Offline evaluation artifact** — seed "
            f"{artifact.get('seed')} · mode `{artifact.get('mode')}`"
        )
        st.caption(
            f"dataset: `{dataset.get('graph_name')}` · alerts requested: "
            f"{dataset.get('alerts_requested')} · bounded subset: "
            f"{bool(dataset.get('bounded_subset'))} · generated "
            f"{artifact.get('generated_at')}"
        )
        st.caption(
            "These numbers describe that offline evaluation run only — they "
            "are NOT this investigation's results and are never re-used as "
            "such."
        )
        rows = vm.eval_metric_rows(artifact)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True, height=280)
        else:
            st.caption("The artifact is identified but has no displayable "
                       "aggregate metrics.")


# ------------------------------------------------------------------- main


def _start_job(row: dict, settings, *, mode: str, mode_label: str) -> None:
    """Start one investigation for this alert (used by both run controls)."""
    st.session_state.run = None
    st.session_state.selected_node = None
    st.session_state.selected_edge = None
    st.session_state.mode_label = mode_label
    st.session_state.polls = 0
    st.session_state.job = backend.start_investigation(
        row["alert_id"], settings=settings,
        mode=mode,
        existing=_job(),
    )


def _render_run_panel(settings, row: dict, *, status: str) -> None:
    st.markdown("#### Run investigation")
    creds = backend.credentials_configured(settings)
    mode_choice = st.radio(
        "LLM mode",
        ["FakeLLM (deterministic)", f"Real provider ({settings.llm_provider})"],
        index=0, key="mode_radio",
        help="Real mode needs credentials already present in the environment; "
             "the UI never collects keys.",
    )
    want_real = mode_choice.startswith("Real")
    if want_real and not creds:
        st.warning("Provider not configured — the run will report a clear "
                   "configuration error. FakeLLM always works.")
    running = status == "running"
    if st.button("Investigate alert", type="primary",
                 width="stretch", disabled=running):
        _start_job(row, settings, mode="real" if want_real else "fake",
                   mode_label=mode_choice)
        st.rerun()
    if running:
        st.caption("Investigation running — the timeline, evidence ledger and "
                   "graph update from real backend events.")

    with st.expander("Demo data (explicit reset)"):
        st.caption("Rebuilds ONLY the disposable demo graph, after "
                   "confirmation — never automatically.")
        confirm = st.checkbox("I confirm: rebuild the demo graph",
                              key="confirm_reset")
        if st.button("Rebuild demo graph", disabled=not confirm):
            with st.spinner("Reseeding the disposable demo graph..."):
                info = backend.reset_demo_graph(settings)
            st.session_state.inbox = None
            _reset_view_state()
            st.success(f"Rebuilt {info['graph_name']} (seed {info['seed']}).")
            st.rerun()


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()
    settings = backend.ui_settings()

    _finalize_job(settings)
    view, partial, events = _current_view()

    # Never show another alert's result: drop a stored run that does not match.
    if st.session_state.run is not None and vm.is_stale(
        st.session_state.run, st.session_state.selected_alert or ""
    ):
        st.session_state.run = None
        view, partial, events = _current_view()

    _render_intro(
        settings,
        alert_selected=st.session_state.selected_alert is not None,
        events=events,
    )

    job = _job()
    status = _run_state(view, partial, job)
    elapsed = job.elapsed_s if job is not None else view.get("elapsed_s")

    left, main_col, right = st.columns([1.15, 2.5, 1.5])

    with left:
        row = _render_inbox(settings)
        if row is not None:
            _render_guided_demo(view, status=status)

    if row is None:
        with main_col:
            st.markdown('<div class="vacant">Load an alert to begin.</div>',
                        unsafe_allow_html=True)
        return

    # A guided example starts here — only after the user pressed its button.
    pending = st.session_state.get("pending_run")
    st.session_state.pending_run = None
    if pending and pending.get("alert_id") == row["alert_id"]:
        _start_job(row, settings, mode="fake",
                   mode_label="FakeLLM (deterministic)")
        st.rerun()

    with right:
        _render_run_panel(settings, row, status=status)
        st.markdown("---")
        _render_timeline(events, status=status, elapsed=elapsed)
        st.markdown("---")
        _render_hypotheses(view, partial=partial)

    with main_col:
        _render_metric_cards(view, events, settings, partial=partial)
        tabs = st.tabs(["Graph & evidence", "Decision & audit", "Technical & evaluation"])
        with tabs[0]:
            _render_graph_panel(settings, row, view, partial=partial)
            st.markdown("---")
            _render_evidence_tab(view, partial=partial)
        with tabs[1]:
            _render_decision(view, partial=partial)
            st.markdown("---")
            _render_audit(view, partial=partial)
        with tabs[2]:
            _render_technical(
                settings, view, events, partial=partial, status=status,
                graph_status=_ensure_graph_status(settings),
            )

    if job is not None and job.running:
        st.session_state.polls += 1
        if st.session_state.polls > MAX_POLLS:
            st.warning("Stopping live updates; the run continues in the "
                       "background and its result will appear when it finishes.")
            return
        time.sleep(POLL_SECONDS)
        st.rerun()


main()