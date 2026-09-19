// Part 2 deliverable: a two-page summary document.
// Regenerate with:  node build_doc.js BL_MLOps_Part2.docx
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  LevelFormat, convertInchesToTwip,
} = require("docx");
const fs = require("fs");

const INK = "151A24";
const SLATE = "2E3B4E";
const MUTED = "6B7787";
const RULE = "D8DDE3";
const HEAD_BG = "EEF1F4";

const BODY = 19;   // half-points => 9.5pt
const H = 22;      // 11pt
const FONT = "Calibri";

const TABLE_W = 10080; // DXA, fits inside 0.6in margins on US Letter

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 75, line: opts.line ?? 208 },
    alignment: opts.align,
    children: [new TextRun({
      text, font: FONT, size: opts.size ?? BODY,
      color: opts.color ?? SLATE, bold: opts.bold, italics: opts.italics,
    })],
  });
}

// A paragraph built from [text, {bold}] pairs, so key figures can be emphasised inline.
function rich(parts, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 75, line: opts.line ?? 208 },
    children: parts.map(([text, o = {}]) => new TextRun({
      text, font: FONT, size: o.size ?? BODY,
      color: o.color ?? SLATE, bold: o.bold, italics: o.italics,
    })),
  });
}

function heading(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 130, after: 60 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 2 } },
    children: [new TextRun({ text, font: FONT, size: H, bold: true, color: INK })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "dash", level: 0 },
    spacing: { after: 34, line: 205 },
    children: [new TextRun({ text, font: FONT, size: BODY, color: SLATE })],
  });
}

function cell(text, { bold, header, width, align } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { type: ShadingType.CLEAR, fill: HEAD_BG, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 90, right: 90 },
    children: [new Paragraph({
      spacing: { after: 0, line: 200 },
      alignment: align,
      children: [new TextRun({
        text, font: FONT, size: BODY - 1,
        color: header ? INK : SLATE, bold: bold || header,
      })],
    })],
  });
}

const COLS = [2450, 2200, 5430];
const BURST_COLS = [1850, 1600, 1700, 1500, 3430];

function table(rows, cols) {
  const widths = cols || COLS;
  return new Table({
    width: { size: TABLE_W, type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 3, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 3, color: RULE },
      left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 3, color: RULE },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    },
    rows: rows.map((r, i) => new TableRow({
      children: r.map((t, j) => cell(t, { header: i === 0, width: widths[j] })),
    })),
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "dash",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 200, hanging: 140 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: {
          top: convertInchesToTwip(0.6), bottom: convertInchesToTwip(0.55),
          left: convertInchesToTwip(0.7), right: convertInchesToTwip(0.7),
        },
      },
    },
    children: [
      new Paragraph({
        spacing: { after: 20 },
        children: [new TextRun({
          text: "Business Loans — Brand Ranking", font: FONT, size: 30, bold: true, color: INK,
        })],
      }),
      new Paragraph({
        spacing: { after: 140 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: INK, space: 3 } },
        children: [new TextRun({
          text: "Productionising two research scripts — implementation, decisions and operation",
          font: FONT, size: BODY + 1, color: MUTED, italics: true,
        })],
      }),

      heading("1. Implementation"),
      rich([
        ["The two research scripts are productionised without altering their functions or logic. The originals are vendored ", {}],
        ["byte-identical", { bold: true }],
        [" and the production code subclasses them, overriding only where the input comes from, where artifacts are written, when models are loaded, and how logging is handled. No feature engineering and no hyper-parameters are touched.", {}],
      ]),
      p("The system comprises versioned Delta ingestion; a training pipeline preserving both researcher modes (train_test evaluates and writes nothing, production trains on all data and registers artifacts); MLflow tracking with 13 parameters, 18 metrics parsed from the researcher's own log, that log uploaded verbatim, and six charts per run; a composite registered model; a FastAPI endpoint; a weekly schedule; and a load-simulation suite."),
      rich([
        ["The registry is the seam between training and serving. Training writes versions; serving resolves the ", {}],
        ["@champion", { bold: true }],
        [" alias at startup. Neither side knows about the other, so the same package runs locally and on Databricks with only the tracking URI and the data location differing — and both arrive as task parameters rather than as code changes.", {}],
      ]),

      heading("2. The central dilemma: a transformer on a synchronous path"),
      rich([
        ["TabPFN is an in-context learner: ", {}],
        ["fit()", { bold: true }],
        [" installs a 1,000-row context rather than training. The original calls it inside the prediction function, so every request reloaded the model, re-uploaded that context over the network and refitted. A direct port measured ", {}],
        ["77,000 ms per request", { bold: true }],
        [" against a user waiting on a landing page.", {}],
      ]),
      rich([
        ["Two decisions resolved it. The context is fitted ", {}],
        ["once at startup", { bold: true }],
        [" instead of per request: because it is a fixed artifact, re-fitting repeats identical work, which I verified is bit-identical. And ", {}],
        ["fit_mode", { bold: true }],
        [" is set to cache the context's transformer state. Together: ", {}],
        ["1,067 ms per request, a 70× reduction", { bold: true }],
        [", with equivalence verified at a maximum absolute difference of 2.75e-4 on payouts averaging $74.77 and identical ranking order. This is a compute optimisation, not a model change.", {}],
      ]),
      rich([
        ["One trade-off was declined. Reducing TabPFN's ensemble size gives a further 9×, and Spearman correlation stays at 0.82, but users click the first brand and ", {}],
        ["the top-ranked brand changes for half of all users", { bold: true }],
        [". That is a different product rather than a latency improvement.", {}],
      ]),

      heading("3. Decisions and alternatives considered"),
      table([
        ["Decision", "Alternative rejected", "Basis"],
        ["Local TabPFN weights as the default", "Hosted Prior Labs API (the original design)", "Both ship and are runtime-selectable. Medians tie, but hosted p99 was 19.3 s against 1.25 s local, it is rate-limited to 60/min, and its endpoint was retired by the provider mid-project. Hosted wins under burst load, so the switch is a one-command overlay."],
        ["A single uvicorn worker", "Two or more workers", "Measured worse: throughput fell from 1.07 to 0.83 rps. One request already parallelises across every core, so a second worker divides the same silicon."],
        ["No dynamic batching", "Batching concurrent users into one call", "The crossover is ~15 simultaneous users; measured peak demand is 11, and within a batch every user waits for the slowest."],
        ["FastAPI over MLflow's scoring server", "mlflow models serve", "The built-in server is a generic DataFrame wrapper: it does not match the dictionary contract, and gives no control over start-up warm-up, which is the entire latency strategy."],
        ["One composite registered model", "Three independently versioned artifacts", "The classifier, payout context and brand universe must agree. Registered separately, “which combination was serving?” becomes unanswerable and rollback means hand-picking compatible versions."],
      ]),

      new Paragraph({ spacing: { after: 60 }, children: [] }),

      heading("4. How it runs in production"),
      rich([
        ["The production run is scheduled ", {}],
        ["weekly, Sunday 05:00 UTC", { bold: true }],
        [" — an APScheduler container locally, and a Databricks Asset Bundle job in the workspace, where both jobs are deployed and have completed successfully. A second job evaluates two hours earlier, so the accuracy report for the week exists before new artifacts are registered. Both deploy paused.", {}],
      ]),
      rich([
        ["Rollback is an alias move plus a restart; nothing is rebuilt or retrained. Each production run promotes the new version and demotes the incumbent to ", {}],
        ["@previous", { bold: true }],
        [", so a rollback target always exists without anyone preparing one. Note that ", {}],
        ["@champion", { bold: true }],
        [" denotes the version currently served, not the newest: a production run moves it forward, a rollback moves it back, so after a rollback the newest version is deliberately not in service. Serving resolves the alias at startup, so changing version needs no redeployment. Exercised in both directions, not only described.", {}],
      ]),
      rich([
        ["Measured on the shipped container: ", {}],
        ["p50 1,054 ms and p99 1,120 ms", { bold: true }],
        [" at concurrency 1 over 200 requests — a tail 6% above the median, the payoff for moving every expensive operation to startup. A sweep from 1 to 100 concurrent users covered 975 requests with ", {}],
        ["zero failures", { bold: true }],
        ["; throughput is flat at ~1 rps throughout, so the service saturates at one request in flight. Replaying all 62,724 real arrivals through that service rate, ", {}],
        ["92% of users never queue and 0.19% exceed three seconds", { bold: true }],
        [".", {}],
      ]),
      rich([
        ["Simultaneous arrivals were measured two ways, because they answer different questions. A ", {}],
        ["closed-loop", { bold: true }],
        [" test holds a fixed number of virtual users, each waiting for its reply before sending again: at 10 concurrent users the median is 9.33 s, the 99th percentile 10.41 s, and nothing fails. But that model cannot produce a backlog, since a slower server also slows the client. Real traffic does not behave that way, so bursts were also fired ", {}],
        ["open-loop", { bold: true }],
        [", on a wall-clock schedule, whether or not the previous burst had finished.", {}],
      ]),
      table([
        ["10 users every", "offered", "completed", "median wait", "behaviour across repeated bursts"],
        ["6 s", "1.67 rps", "0.96 rps", "50.1 s", "Backlog grows without bound: 17.7 s on the first burst, 60 s by the seventh."],
        ["10 s", "1.00 rps", "0.93 rps", "14.8 s", "Marginally over capacity; the wait creeps from 12.2 s to 19.5 s."],
        ["12 s", "0.83 rps", "0.83 rps", "10.6 s", "Flat across every burst — the queue fully drains between them."],
      ], BURST_COLS),
      rich([
        ["The sustainable spacing is therefore about ", {}],
        ["10.7 s between bursts of ten", { bold: true }],
        [". That threshold, not average throughput, is what should carry an alert: past it the service stops recovering between bursts. Measured demand sits inside it — the busiest second in 62 days held 11 arrivals, but such seconds are isolated rather than repeated. The hosted backend absorbs the same bursts at a 2.3 s median, because it waits on a network in parallel where the local path computes in series.", {}],
      ]),
      p("Cold start is ~130 s, almost entirely the context fit. Deployments must therefore wait for readiness rather than cutting over on process start, and scaling has to be proactive: by the time a spike is visible, replacement capacity is two minutes away."),

      heading("5. Potential issues"),
      bullet("Promotion is unconditional, and this is the highest-priority gap. The 03:00 job does evaluate on a held-out week and logs the result; the 05:00 job neither reads it nor compares against the incumbent. It trains on all data — which by the researcher's mode semantics leaves no held-out set to score the promoted artifact against — registers, and moves the alias. The evidence is produced and then not acted on, so a poor week becomes champion unless a human reads the metrics."),
      bullet("Platform-dependent feature corruption. astype(int) resolves to int32 on Windows, so a ten-digit phone number overflows and a model feature is silently wrong; P(lead) moved from 0.7771 to 0.3632 for the same user. Fixed in the serving layer and reported upstream rather than edited in the researcher's files."),
      bullet("Registered versions pin the environment they were logged with, and those pins have already drifted — one now returns a 503 after the provider retired an endpoint. Harmless to the shipped stack, which builds from its own requirements, but it undermines what a registry is meant to guarantee."),
      bullet("Repeated bursts spaced closer than ~10.7 s do not drain (section 4). This is a capacity limit rather than a defect, but it is the condition to monitor, and current demand gives little warning before it is reached."),
      bullet("client_name carries 40.8% of the classifier's feature importance, which suggests the ranking may be closer to a global brand ordering than to per-user personalisation."),

      heading("6. Ways to improve it"),
      bullet("A promotion gate: have the 05:00 run read the 03:00 evaluation and refuse to move the alias when weighted F1 or payout error regresses beyond a threshold against the incumbent's recorded metrics. The evaluation already exists and the metrics are already logged per run, so this is wiring rather than new machinery, and it is what makes the schedule safe to leave unattended."),
      bullet("A GPU instance. Roughly 90% of the ~1 s is a single TabPFN forward pass, and this is the only lever that moves that floor. It could not be measured here — the development machine has an AMD GPU and Free Edition serving is CPU-only — so it is a reasoned recommendation, not a measured result."),
      bullet("Shadow or A/B evaluation on revenue per session, which is the metric that actually matters rather than MAPE."),
      bullet("Distillation into a fast surrogate if latency must fall further. Not done deliberately: it changes the model, making it a product decision."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = process.argv[2] || "BL_MLOps_Part2.docx";
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
