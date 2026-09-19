"""The service landing page.

Kept out of `app.py` so the route stays one line and the markup is editable on its own.

It exists because opening the service root previously returned `{"detail":"Not Found"}` --
correct, since no route was defined at `/`, but it reads like a broken deployment, and it
is the first thing anyone sees when they click the port in Docker Desktop.

TEMPLATING: placeholder substitution, NOT an f-string
----------------------------------------------------
The page carries JavaScript, and JS is full of `{`, `}` and backslash escapes. In an
f-string every brace has to be doubled and every backslash survives into Python's escape
handling -- which silently corrupted this file twice: a `\\n` inside a JS string literal
became a real newline, producing a syntax error that killed the whole script and made a
click navigate instead of running its handler.

So the markup is a plain string with `__TOKEN__` placeholders, substituted by `render()`.
Braces and escapes in the JS are then exactly what the browser receives.
"""
from __future__ import annotations

import json
from typing import Any

# Project palette: one hue for structure, amber for the live signal, teal for ready.
_CSS = """
:root{
  --bg:#f7f8f9; --surface:#ffffff; --ink:#151a24; --muted:#6b7787; --rule:#e3e7ec;
  --accent:#2a78d6; --ok:#2f7d6e; --ok-bg:#e8f4f0; --wait:#8a5a00; --wait-bg:#fdf3e3;
  --code-bg:#f2f4f6; --bar:#cde2fb;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#14161a; --surface:#1b1e24; --ink:#eef1f5; --muted:#9aa4b2; --rule:#2a2f38;
    --accent:#5d9ded; --ok:#5fb8a4; --ok-bg:#16302b; --wait:#e0a34a; --wait-bg:#33260f;
    --code-bg:#22262e; --bar:#1e3a5c;
  }
}
*{box-sizing:border-box}
body{
  margin:0; padding:3rem 1.5rem; background:var(--bg); color:var(--ink);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
}
.wrap{max-width:52rem;margin:0 auto}
header{display:flex;align-items:flex-start;justify-content:space-between;gap:1.5rem;
  flex-wrap:wrap;margin-bottom:.35rem}
h1{font-size:1.45rem;font-weight:650;letter-spacing:-.01em;margin:0}
.tagline{color:var(--muted);margin:.15rem 0 2rem}
.pill{display:inline-flex;align-items:center;gap:.45rem;padding:.3rem .75rem;
  border-radius:999px;font-size:.8rem;font-weight:600;white-space:nowrap}
.pill.ready{background:var(--ok-bg);color:var(--ok)}
.pill.wait{background:var(--wait-bg);color:var(--wait)}
.dot{width:.5rem;height:.5rem;border-radius:50%;background:currentColor}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:.75rem;
  margin-bottom:2rem}
.fact{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:.85rem 1rem}
.fact .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.fact .v{font-size:1.15rem;font-weight:650;margin-top:.15rem;
  font-variant-numeric:tabular-nums}
h2{font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
  font-weight:650;margin:0 0 .7rem}
.links{display:grid;gap:.6rem;margin-bottom:2rem}
a.card{display:flex;align-items:center;gap:.9rem;background:var(--surface);
  border:1px solid var(--rule);border-radius:10px;padding:.85rem 1rem;
  text-decoration:none;color:inherit;transition:border-color .12s,transform .12s}
a.card:hover{border-color:var(--accent);transform:translateY(-1px)}
.verb{flex:none;font:600 .7rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  padding:.35rem .5rem;border-radius:5px;background:var(--code-bg);color:var(--accent);
  letter-spacing:.03em}
.card .t{font-weight:600}
.card .d{color:var(--muted);font-size:.87rem}
.hint{color:var(--muted);font-size:.8rem;margin:.5rem 0 0}
.card .arrow{margin-left:auto;color:var(--muted)}
pre{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:.9rem 1rem;overflow-x:auto;margin:0;
  font:.82rem/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink)}
footer{margin-top:2.25rem;padding-top:1.25rem;border-top:1px solid var(--rule);
  color:var(--muted);font-size:.83rem}

/* --- live ranking result ------------------------------------------------------- */
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:1rem 1.1rem;margin:-1.4rem 0 2rem}
.panel .meta{color:var(--muted);font-size:.8rem;
  font-variant-numeric:tabular-nums;margin-bottom:.35rem}
.panel .note{color:var(--muted);font-size:.78rem;margin:.7rem 0 0}
table.rank{width:100%;border-collapse:collapse;font-size:.88rem}
table.rank th{text-align:left;font-weight:650;color:var(--muted);font-size:.68rem;
  text-transform:uppercase;letter-spacing:.06em;padding:.35rem .5rem;
  border-bottom:1px solid var(--rule)}
table.rank th.num,table.rank td.num{text-align:right;font-variant-numeric:tabular-nums}
table.rank td{padding:.4rem .5rem;border-bottom:1px solid var(--rule)}
table.rank tr:last-child td{border-bottom:0}
table.rank td.pos{color:var(--muted);width:2.2rem;font-variant-numeric:tabular-nums}
table.rank tr.top td{font-weight:650}
.brandcell{position:relative;display:block}
.brandcell .bar{position:absolute;inset:0 auto 0 0;background:var(--bar);
  border-radius:3px;z-index:0}
.brandcell .name{position:relative;z-index:1;padding:0 .3rem}
"""

# `__TOKEN__` placeholders are substituted in `render()`. See the module docstring for
# why this is not an f-string. It is also a RAW string: an ordinary Python string still
# processes escape sequences, so an escaped newline written inside the JavaScript would
# become a real line break and split the string literal it sits in -- a syntax error that
# silently disables the whole script. The r-prefix passes escapes through untouched.
_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BL Brand Ranking</title><style>__CSS__</style></head>
<body><div class="wrap">

<header>
  <div>
    <h1>Business Loans &mdash; Brand Ranking</h1>
    <p class="tagline">Ranks lender brands for one user by expected payout.</p>
  </div>
  <span class="pill __STATE_CLS__"><span class="dot"></span>__STATE_TXT__</span>
</header>

<div class="facts">
  <div class="fact"><div class="k">Model version</div><div class="v">__VERSION__</div></div>
  <div class="fact"><div class="k">TabPFN</div><div class="v">__TABPFN__</div></div>
  <div class="fact"><div class="k">Brand universe</div><div class="v">__BRANDS__</div></div>
  <div class="fact"><div class="k">Typical latency</div><div class="v">~1.0 s</div></div>
</div>

<h2>Endpoints</h2>
<div class="links">
  <a class="card" id="rank-card" href="__RANK_HREF__">
    <span class="verb">POST</span>
    <span><span class="t">/rank</span><br><span class="d">One user's post-funnel data in,
      ranked brands out &mdash; <strong>click to run it here</strong> on the example
      user</span></span>
    <span class="arrow">&rarr;</span>
  </a>
  <a class="card" href="/docs">
    <span class="verb">UI</span>
    <span><span class="t">/docs</span><br><span class="d">The full API reference: every
      endpoint, its schema and its responses</span></span>
    <span class="arrow">&rarr;</span>
  </a>
  <a class="card" id="health-card" href="/health">
    <span class="verb">GET</span>
    <span><span class="t">/health</span><br><span class="d">Readiness and the live model
      version &mdash; shown here, without leaving the page</span></span>
    <span class="arrow">&rarr;</span>
  </a>
  <a class="card" href="http://localhost:5000" target="_blank" rel="noopener">
    <span class="verb">MLF</span>
    <span><span class="t">MLflow</span><br><span class="d">Runs, metrics, charts and the
      model registry</span></span>
    <span class="arrow">&rarr;</span>
  </a>
</div>

<div class="panel" id="rank-out" hidden></div>
<pre id="health-out" hidden></pre>

<h2>From the command line</h2>
<pre>curl -s -X POST localhost:8088/rank -H "content-type: application/json" -d @examples/user.json</pre>
<p class="hint">On Windows PowerShell use <code>curl.exe</code> &mdash; <code>curl</code>
there is an alias for <code>Invoke-WebRequest</code>, which rejects these flags.</p>

<footer>
  Serving <code>__MODEL_URI__</code>. The model is loaded once at startup and the TabPFN
  context is fitted there too, which is why a cold start takes ~130&nbsp;s and a request
  takes ~1&nbsp;s.
</footer>

</div>

<script type="application/json" id="example-user">__EXAMPLE_USER__</script>
<script>
(function () {
  // Both handlers keep their card's href, so if a listener fails to attach the card still
  // works as an ordinary link. Nothing here is required for the endpoint to function.
  var example = null;
  try {
    example = JSON.parse(document.getElementById('example-user').textContent);
  } catch (err) {
    example = null;
  }

  function money(v) {
    return '$' + Number(v).toFixed(2);
  }

  // Built with DOM calls rather than innerHTML: brand names come from the model, and
  // string-concatenated markup would be an injection point.
  function renderRanking(data, wallMs) {
    var panel = document.getElementById('rank-out');
    panel.hidden = false;
    panel.textContent = '';

    var entries = Object.keys(data.ranking).map(function (name) {
      return { name: name, payout: data.ranking[name].expected_payout };
    });
    entries.sort(function (a, b) { return b.payout - a.payout; });

    var meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent =
      'model v' + data.model_version +
      '  ·  ' + entries.length + ' brands scored' +
      '  ·  ' + Math.round(data.latency_ms) + ' ms model time' +
      '  ·  ' + Math.round(wallMs) + ' ms round trip';
    panel.appendChild(meta);

    if (entries.length === 0) {
      var empty = document.createElement('p');
      empty.className = 'note';
      empty.textContent =
        'No ranking for this user - preprocessing dropped the row, which the endpoint ' +
        'reports as an empty ranking rather than an error.';
      panel.appendChild(empty);
      return;
    }

    var max = entries[0].payout || 1;
    var table = document.createElement('table');
    table.className = 'rank';

    var head = document.createElement('tr');
    ['', 'Brand', 'Expected payout'].forEach(function (label, i) {
      var th = document.createElement('th');
      th.textContent = label;
      if (i === 2) th.className = 'num';
      head.appendChild(th);
    });
    table.appendChild(head);

    entries.forEach(function (row, i) {
      var tr = document.createElement('tr');
      if (i === 0) tr.className = 'top';

      var pos = document.createElement('td');
      pos.className = 'pos';
      pos.textContent = String(i + 1);
      tr.appendChild(pos);

      var brand = document.createElement('td');
      var cell = document.createElement('span');
      cell.className = 'brandcell';
      var bar = document.createElement('span');
      bar.className = 'bar';
      // Proportional to the top brand, so the spread between positions is visible.
      bar.style.width = Math.max(2, (row.payout / max) * 100) + '%';
      var name = document.createElement('span');
      name.className = 'name';
      name.textContent = row.name;
      cell.appendChild(bar);
      cell.appendChild(name);
      brand.appendChild(cell);
      tr.appendChild(brand);

      var payout = document.createElement('td');
      payout.className = 'num';
      payout.textContent = money(row.payout);
      tr.appendChild(payout);

      table.appendChild(tr);
    });
    panel.appendChild(table);

    var note = document.createElement('p');
    note.className = 'note';
    note.textContent =
      'One real POST /rank for the example user. Bar length is relative to the ' +
      'top brand. Full schema and a request editor are in the API explorer.';
    panel.appendChild(note);
  }

  var rankCard = document.getElementById('rank-card');
  if (rankCard && example) {
    rankCard.addEventListener('click', function (e) {
      e.preventDefault();
      var panel = document.getElementById('rank-out');
      panel.hidden = false;
      panel.textContent = 'ranking the example user ... (about a second)';
      var started = performance.now();
      fetch('/rank', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(example)
      })
        .then(function (r) {
          if (!r.ok) {
            return r.text().then(function (t) {
              throw new Error('HTTP ' + r.status + ' - ' + t);
            });
          }
          return r.json();
        })
        .then(function (data) { renderRanking(data, performance.now() - started); })
        .catch(function (err) {
          panel.textContent = 'request failed: ' + err.message +
            ' - the service may still be warming up (~130s from a cold start).';
        });
    });
  }

  var healthCard = document.getElementById('health-card');
  var healthOut = document.getElementById('health-out');
  if (healthCard && healthOut) {
    healthCard.addEventListener('click', function (e) {
      e.preventDefault();
      healthOut.hidden = false;
      healthOut.textContent = 'requesting /health ...';
      fetch('/health')
        .then(function (r) {
          return r.json().then(function (j) { return { status: r.status, body: j }; });
        })
        .then(function (d) {
          healthOut.textContent =
            'HTTP ' + d.status + '\n\n' + JSON.stringify(d.body, null, 2);
        })
        .catch(function (err) { healthOut.textContent = 'request failed: ' + err; });
    });
  }
})();
</script>

</body></html>"""


def render(
    version: str,
    loaded: bool,
    tabpfn: str,
    model_uri: str,
    brands: int | None = None,
    example_user: dict[str, Any] | None = None,
) -> str:
    """Render the landing page.

    `example_user` is embedded so the page can run a real ranking without a second
    round-trip to fetch a payload, and without adding an endpoint that exists only to
    serve the demo.
    """
    state_cls, state_txt = ("ready", "ready") if loaded else ("wait", "loading (~130s)")
    # Fallback for the /rank card when JavaScript is unavailable: POST cannot be a
    # hyperlink, so the href points at the operation inside the Swagger UI.
    rank_href = "/docs#/default/rank_rank_post"

    # `</script>` inside embedded JSON would close the tag early; escaping the slash keeps
    # the payload valid JSON while making that impossible.
    example_json = json.dumps(example_user or {}).replace("</", "<\\/")

    tokens = {
        "__CSS__": _CSS,
        "__STATE_CLS__": state_cls,
        "__STATE_TXT__": state_txt,
        "__VERSION__": str(version),
        "__TABPFN__": tabpfn,
        "__BRANDS__": str(brands) if brands is not None else "&mdash;",
        "__RANK_HREF__": rank_href,
        "__MODEL_URI__": model_uri,
        "__EXAMPLE_USER__": example_json,
    }
    html = _HTML
    for token, value in tokens.items():
        html = html.replace(token, value)
    return html
