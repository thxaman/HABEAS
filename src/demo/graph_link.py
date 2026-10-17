"""FalkorDB Browser link helper for the demo (Phase 8).

Honesty contract (Phase 8 spec §3): never fabricate a query parameter or a
deep-link format. The FalkorDB Browser was NOT installed in the development
environment (``http://localhost:3000`` was unreachable — see
``docs/phase-8-demo.md``), so no deep-link URL format could be verified
against a real instance. Rather than invent one, this helper only ever prints:

- the explicit local browser base URL,
- the graph name to select, and
- a copyable Cypher query (optional).

``GraphLink.deep_link`` is therefore always ``None`` and is documented as
intentional: graph visualization is supplementary and never feeds the
investigation or the decision, which are grounded in actual FalkorDB query
results. If a browser is later detected the format is still not asserted,
because it was not verified here.
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

# Override for a remote/renamed browser instance.
DEFAULT_BROWSER_URL = "http://localhost:3000"


@dataclass(frozen=True)
class GraphLink:
    """Everything the demo prints so a human can open the graph in a browser."""

    browser_url: str
    graph_name: str
    cypher: str
    reachable: bool | None
    # Always None: no deep-link format was verifiable, so none is emitted.
    deep_link: str | None = None

    def render(self) -> str:
        lines = [
            f"Graph browser: {self.browser_url}",
            f"  select graph: {self.graph_name}",
        ]
        if self.reachable is False:
            lines.append(
                "  note: no FalkorDB Browser answered at that address, so the "
                "link is informational only"
            )
        elif self.reachable is True:
            lines.append("  note: a browser answered; explore the subgraph manually")
        else:
            lines.append("  note: browser reachability not probed")
        lines.append(
            "  deep link: not provided — the browser's URL/deep-link format "
            "was not verified in this environment, so none is fabricated"
        )
        if self.cypher:
            lines.extend(["", "  copyable Cypher for the subgraph:", self.cypher])
        return "\n".join(lines)


def browser_url() -> str:
    return os.getenv("HABEAS_BROWSER_URL", DEFAULT_BROWSER_URL).rstrip("/")


def _probe(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_browser(url: str | None = None, *, timeout: float = 0.5) -> bool:
    """True when something is listening on the browser's host/port."""
    parsed = urlparse(url or browser_url())
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return _probe(host, port, timeout=timeout)


def build_graph_link(
    graph_name: str,
    cypher: str = "",
    *,
    url: str | None = None,
    probe: bool = True,
) -> GraphLink:
    """Build the printed browser hint. Never returns a deep link (see module)."""
    target = url or browser_url()
    reachable = probe_browser(target) if probe else None
    return GraphLink(
        browser_url=target,
        graph_name=graph_name,
        cypher=cypher.strip(),
        reachable=reachable,
        deep_link=None,
    )


def subgraph_cypher(account_ids: list[str], *, limit: int = 200) -> str:
    """A bounded, parameter-free-friendly read query for a focal subgraph.

    Only account ids that the demo already holds are interpolated, and only
    after validation by the caller is expected to pass graph ids. The query is
    for a human to paste into a browser; it is never executed by an agent.
    """
    ids = [a for a in account_ids if a]
    listing = ", ".join(f"'{a}'" for a in ids)
    return (
        f"MATCH p = (a:Account)-[:TRANSFERRED_TO]->(b:Account)\n"
        f"WHERE a.id IN [{listing}] AND b.id IN [{listing}]\n"
        f"RETURN p LIMIT {int(limit)}"
    )