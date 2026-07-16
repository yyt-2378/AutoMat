"""Small browser demo for the agent_qwen harness.

Run:
    python -m agent_qwen.demo_server --port 8765
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .defaults import PIPELINE_DEFAULTS
from .harness import AgentQwenHarness
from .request_intake import normalize_harness_request


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>agent_qwen STEM Agent Workflow</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101312;
      --panel: #181d1b;
      --line: #303a35;
      --text: #edf3ef;
      --muted: #a8b5ae;
      --accent: #5dd39e;
      --bad: #ff8f8f;
      --warn: #f2c36b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { max-width: 1220px; margin: 0 auto; padding: 28px; }
    header { display: flex; justify-content: space-between; gap: 20px; align-items: end; margin-bottom: 18px; }
    h1 { margin: 0; font-size: 31px; letter-spacing: 0; }
    p { color: var(--muted); line-height: 1.5; margin: 8px 0 0; }
    .pill { color: var(--accent); border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; white-space: nowrap; }
    .layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); gap: 16px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .panel h2 { margin: 0; font-size: 15px; padding: 13px 14px; border-bottom: 1px solid var(--line); }
    form { padding: 14px; display: grid; gap: 12px; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; }
    input, select, textarea, button {
      width: 100%;
      border: 1px solid var(--line);
      background: #0d100f;
      color: var(--text);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
    }
    textarea { min-height: 82px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .check { display: flex; align-items: center; gap: 9px; color: var(--text); }
    .check input { width: auto; }
    button { background: var(--accent); border-color: var(--accent); color: #07100b; font-weight: 750; cursor: pointer; }
    button:disabled { opacity: .65; cursor: wait; }
    .summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; padding: 14px; border-bottom: 1px solid var(--line); }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-height: 76px; }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric strong { display: block; margin-top: 8px; font-size: 20px; }
    .steps { padding: 14px; display: grid; gap: 10px; }
    .step { border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
    .step-top { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .ok { color: var(--accent); font-weight: 700; }
    .fail { color: var(--bad); font-weight: 700; }
    code { color: var(--muted); word-break: break-all; }
    pre { margin: 12px 0 0; padding: 12px; background: #0d100f; border-radius: 6px; overflow: auto; color: #d6e6dc; max-height: 260px; }
    .empty { padding: 20px; color: var(--muted); }
    @media (max-width: 900px) {
      main { padding: 18px; }
      header, .layout { display: block; }
      .pill { display: inline-block; margin-top: 12px; }
      .panel { margin-top: 14px; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>agent_qwen STEM Agent Workflow</h1>
      <p>Run natural-language requests through intake, then execute explicit harness lines: classic, direct, or compare.</p>
    </div>
    <div class="pill">harness demo</div>
  </header>
  <section class="layout">
    <article class="panel">
      <h2>Run</h2>
      <form id="run-form">
        <label>Harness line
          <select name="harness_line">
            <option value="classic">classic</option>
            <option value="direct">direct</option>
            <option value="compare">compare</option>
          </select>
        </label>
        <label>Image path
          <input name="image_path" value="examples/stem.png">
        </label>
        <label>Denoised image path
          <input name="denoised_img" value="examples/denoised.png">
        </label>
        <label>Elements
          <input name="elements" value="Mo S">
        </label>
        <label>User message
          <textarea name="user_message">元素: Mo S，请走 classic harness line 并给出不确定性。</textarea>
        </label>
        <label>Work root
          <input name="work_root" value="__DEFAULT_WORK_ROOT__">
        </label>
        <div class="row">
          <label class="check"><input name="dry_run" type="checkbox"> dry-run</label>
          <label class="check"><input name="run_confidence" type="checkbox" checked> confidence</label>
        </div>
        <label class="check"><input name="skip_property" type="checkbox" checked> skip property</label>
        <button id="run-btn" type="submit">Run harness line</button>
      </form>
    </article>
    <article class="panel">
      <h2>Result</h2>
      <div id="result"><div class="empty">No run yet.</div></div>
    </article>
  </section>
</main>
<script>
const form = document.getElementById('run-form');
const btn = document.getElementById('run-btn');
const result = document.getElementById('result');

function elementsFrom(text) {
  return text.split(/[ ,，\\n\\t]+/).map(x => x.trim()).filter(Boolean);
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function render(payload) {
  if (!payload.ok) {
    const guidance = (payload.guidance || []).map(item => `<p>${esc(item)}</p>`).join('');
    result.innerHTML = `<div class="empty"><span class="fail">${payload.need_input ? 'Need input' : 'Failed'}</span>${guidance}<pre>${esc(JSON.stringify(payload, null, 2))}</pre></div>`;
    return;
  }
  const run = payload.result || payload;
  const artifacts = run.artifacts || {};
  const steps = run.steps || [];
  result.innerHTML = `
    <div class="summary">
      <div class="metric"><span>Status</span><strong>${run.ok ? 'OK' : 'Failed'}</strong></div>
      <div class="metric"><span>Harness Line</span><strong>${esc(run.harness_line || run.workflow || '-')}</strong></div>
      <div class="metric"><span>Steps</span><strong>${steps.length}</strong></div>
      <div class="metric"><span>Artifacts</span><strong>${Object.keys(artifacts).length}</strong></div>
    </div>
    <div class="steps">
      ${steps.map((step, idx) => `
        <div class="step">
          <div class="step-top">
            <strong>${idx + 1}. ${esc(step.skill)}</strong>
            <span class="${step.success ? 'ok' : 'fail'}">${step.success ? 'OK' : 'FAILED'}</span>
          </div>
          <p>${esc(step.message || '')}</p>
          ${Object.entries((step.result && step.result.artifacts) || {}).map(([k, v]) => `<div><code>${esc(k)}: ${esc(v)}</code></div>`).join('')}
          <pre>${esc(JSON.stringify(step.result || {}, null, 2))}</pre>
        </div>
      `).join('')}
    </div>`;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  btn.disabled = true;
  btn.textContent = 'Running...';
  const data = new FormData(form);
  const payload = {
    harness_line: data.get('harness_line'),
    image_path: data.get('image_path'),
    denoised_img: data.get('denoised_img'),
    elements: elementsFrom(data.get('elements')),
    user_message: data.get('user_message'),
    work_root: data.get('work_root'),
    dry_run: data.get('dry_run') === 'on',
    run_confidence: data.get('run_confidence') === 'on',
    skip_property: data.get('skip_property') === 'on'
  };
  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify(payload)
    });
    render(await response.json());
  } catch (err) {
    render({ok: false, error: String(err)});
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run harness line';
  }
});
</script>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "AgentQwenHarnessDemo/0.1"

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        body = HTML.replace("__DEFAULT_WORK_ROOT__", PIPELINE_DEFAULTS["work_root"]).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            req = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            intake = normalize_harness_request(
                req.get("user_message", ""),
                req,
                default_dry_run=bool(req.get("dry_run", True)),
            )
            if not intake.ready:
                self._send_json({"agent_id": "characterization_recon_agent", **intake.need_input_payload()})
                return
            work_root = intake.work_root or PIPELINE_DEFAULTS["work_root"]
            harness = AgentQwenHarness.from_defaults(
                work_root=work_root,
                dry_run=intake.dry_run,
            )
            if intake.harness_line == "direct":
                run = harness.run_direct_workflow(
                    denoised_img=intake.denoised_img or "",
                    elements=intake.elements,
                    run_property=not intake.skip_property,
                    run_confidence=intake.run_confidence,
                )
            elif intake.harness_line == "compare":
                run = harness.run_line(
                    "compare",
                    image_path=intake.image_path or "",
                    user_message=intake.user_message,
                    elements=intake.elements,
                    run_property=not intake.skip_property,
                    run_confidence=intake.run_confidence,
                )
            else:
                run = harness.run_classic_workflow(
                    image_path=intake.image_path or "",
                    user_message=intake.user_message,
                    elements=intake.elements,
                    run_property=not intake.skip_property,
                    run_confidence=intake.run_confidence,
                )
            self._send_json({"ok": run.ok, "intake": intake.as_context(), "result": run.__dict__})
        except Exception as exc:
            self._send_json({"ok": False, "error": repr(exc)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the agent_qwen harness browser demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Serving agent_qwen harness demo on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
