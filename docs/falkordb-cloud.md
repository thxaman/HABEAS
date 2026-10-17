# FalkorDB Cloud deployment (Streamlit Community Cloud)

HABEAS runs unchanged against a managed FalkorDB Cloud instance: the same
template-only queries, the same MCP boundary, the same Policy v2 gates. This
document covers the connection settings, the one-time graph seed, and the
Streamlit Community Cloud configuration. It contains no real credentials —
everywhere below, `<...>` placeholders must be replaced with YOUR values,
entered only in `.env` (never committed) or Streamlit Cloud secrets.

## 1. Connection settings

| Variable | Meaning | Default (local Docker) |
| --- | --- | --- |
| `FALKORDB_HOST` | Database host | `localhost` |
| `FALKORDB_PORT` | Database port | `6379` |
| `FALKORDB_USERNAME` | Username (empty = no auth) | empty |
| `FALKORDB_PASSWORD` | Password, held as a `SecretStr`; never logged, never rendered | empty |
| `FALKORDB_TLS` | `true` to speak TLS (required by FalkorDB Cloud) | `false` |
| `GRAPH_NAME` | Graph the tools serve | `HABEAS_demo_local` |

Leaving username/password empty and `FALKORDB_TLS=false` reproduces today's
localhost Docker behaviour exactly. The password is stored as a
`pydantic.SecretStr`: `repr()`/`str()` show `**********`, and
`src.db.redact_secrets` scrubs it out of exception messages and log text.

The DB client cache in `src/db.py` is keyed by ALL connection settings
(host, port, username, password, TLS), so two configurations never share a
client, and `db.clear_client_cache()` drops them explicitly.

## 2. One-time seed of the demo graph (explicit, never automatic)

The UI's disposable demo graph is `HABEAS_demo_showcase_s42`. The cloud
database starts empty, so seed it ONCE from a machine that has the
credentials, with the same guarded seeder used locally (it refuses non-demo
graph names, and `--reset-demo-graph` is required to clear an existing graph):

```bash
# bash/linux — run once; the same command re-run is refused unless the graph is empty
FALKORDB_HOST=<your-host>.falke.app \
FALKORDB_PORT=6379 \
FALKORDB_USERNAME=<username> \
FALKORDB_PASSWORD='<password>' \
FALKORDB_TLS=true \
GRAPH_NAME=HABEAS_demo_showcase_s42 \
HABEAS_DEMO_MODE=true \
python -m src.seed.generate --seed 42 --reset-demo-graph
```

Windows PowerShell equivalent:

```powershell
$env:FALKORDB_HOST="<your-host>.falke.app"; $env:FALKORDB_PORT="6379"
$env:FALKORDB_USERNAME="<username>"; $env:FALKORDB_PASSWORD="<password>"
$env:FALKORDB_TLS="true"; $env:GRAPH_NAME="HABEAS_demo_showcase_s42"
$env:HABEAS_DEMO_MODE="true"
python -m src.seed.generate --seed 42 --reset-demo-graph
```

Notes:

- This is a deliberate, human-invoked command. **Nothing in the app reseeds
  the graph on startup, on a rerun, or on a page load** — the UI's only reset
  is the explicit, confirmed control, and it targets the disposable
  `HABEAS_demo_showcase*` namespace only.
- Keep `--seed 42` so the graph matches the committed ground-truth manifest
  the evaluation layer expects.
- `FALKORDB_PASSWORD` is passed through the environment for this one command;
  avoid shell history by leading the assignment with a space (bash `HISTCONTROL`)
  or by using a throwaway shell. It is never written to any file by HABEAS.

## 3. Streamlit Community Cloud configuration

In the Streamlit Cloud dashboard, open the app → **Settings → Secrets** and add
(placeholder values replaced by yours; this file lives only on Streamlit's
servers):

```toml
FALKORDB_HOST = "<your-host>.falke.app"
FALKORDB_PORT = "6379"
FALKORDB_USERNAME = "<username>"
FALKORDB_PASSWORD = "<password>"
FALKORDB_TLS = "true"
GRAPH_NAME = "HABEAS_demo_showcase_s42"
HABEAS_DEMO_MODE = "true"
```

How the secrets reach the graph:

1. `src/ui/backend.py:load_streamlit_secrets_into_env()` promotes these keys
   from `st.secrets` into the app process environment (existing environment
   variables win; a no-op outside Streamlit).
2. `src.config.load_settings()` resolves them into a `Settings` (password as
   `SecretStr`).
3. `src/agent/mcp_client.py:server_parameters()` explicitly forwards the
   resolved `FALKORDB_USERNAME` / `FALKORDB_PASSWORD` / `FALKORDB_TLS` (plus
   host/port/graph) into the MCP server subprocess environment, so the child
   connects to the same cloud database. Values travel in the process
   environment only — HABEAS never logs them.

LLM provider keys (e.g. `GROQ_API_KEY`) may be added to the same secrets
block; the UI never collects or displays them, and `LLM_MODE=real` is only
used when the analyst explicitly switches the mode selector.

## 4. What does NOT change

- All agent graph access still goes through MCP with approved, bounded,
  parameterized templates; no free-form Cypher exists anywhere in the app.
- The write layer is still the sandbox-only `freeze_account` simulation on the
  disposable demo graph, gated by Policy v2, with post-action independent
  verification.
- Historical reports in `reports/` are untouched; UI reports still go to
  `reports/ui/`.
- No ground-truth manifest is ever placed in the agent-visible graph.

## 5. Verifying a cloud deployment

```bash
FALKORDB_HOST=... FALKORDB_USERNAME=... FALKORDB_PASSWORD=... FALKORDB_TLS=true \
GRAPH_NAME=HABEAS_demo_showcase_s42 python -m src.probes
```

The probes exercise the real connection (read templates, hop bounds, timeouts)
against the configured database. Any failure raises a classified
`QueryError` whose message has the password scrubbed.
