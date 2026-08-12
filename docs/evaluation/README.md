# Verified actual-UI evaluation

The workbook in this folder is the current professor-review artifact:

- `NBA-Actual-UI-Agent-Evaluation-Shared-10-Games.xlsx`
- `verified-actual-ui-results.csv` is the Git-diffable export of the exact 30
  model rows embedded in the workbook
- `manifest.json` records the workbook SHA-256, execution contract, and caveats
- 10 identical games for Models A, B, and C
- fixed knowledge cutoff of 2026-04-05
- Model A executed through `POST /api/predict`
- Models B and C executed through the website's `POST /api/run` SSE path
- Model B used `gemma4:latest` and the five required retrieval calls
- Model C used the same language model and retrieval calls, plus Model A's
  predictor as an additional tool
- every required B/C gate receipt passed

The workbook is a manually assembled presentation artifact. It includes
formula-linked Summary, Model A, Model B, Model C, UI Trace, and Methodology
sheets. Its verified result is A 5/10, B 4/10, and C 6/10. This sample supports
a shared classroom comparison only. It does not establish full-season B/C
accuracy.

The agent narratives may cite records and ratings from the prior completed
season because that is the date-gated historical context available to the
agent. Those figures are not current-season final records and should not be
read as post-cutoff information.

To reproduce the underlying actual-UI result CSV, build the web client and run
the bridge in one terminal, then run the evaluator in a second terminal:

```bash
cd ui/web
npm install
npm run build
cd ../..
python -m ui.serve
```

```bash
python -m eval.ui_agent_eval --arms A,B,C \
  --checkpoint eval/results_actual_ui_10.jsonl \
  --out eval/results_actual_ui_10.csv
```

The evaluator requires the local `gemma4:latest` Ollama model for Models B and
C. Evaluation checkpoints and raw result CSVs are ignored because they are
machine-specific runtime outputs. The committed CSV is the bounded,
professor-facing table exported from the reviewed workbook.

The older full-season workbook is intentionally excluded because its B/C rows
were produced before the actual-UI evaluation contract was corrected. A new
full-season workbook should be generated only after every current Model B and C
UI run is complete.
