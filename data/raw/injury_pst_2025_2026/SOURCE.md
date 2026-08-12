# Injury log, 2025-01-13 onward

Continues `data/raw/injury_data_2016_2025/injury_data.csv`, which stops at
2025-01-12 and therefore covered none of the 2025-26 season we replay.

- **Source:** ProSportsTransactions.com basketball transaction search, injury
  ("IL") filter — the same upstream log the Kaggle set was built from, so the
  columns are identical (`Date, Team, Acquired, Relinquished, Notes`) and the
  team names are the same nicknames `agent/teams.py` already maps.
- **Retrieved:** 2026-07-28
- **Range:** 2025-01-13 to 2026-05-29 (3,581 rows). Zero date overlap with the
  Kaggle file, so the two concatenate without dedupe.
- **Read by:** `agent/sources.py` → `INJURY_CSVS`, which loads both files.

## Reproducing this

Not scriptable from the command line: the site returns HTTP 403 to `curl` and
to `requests` regardless of headers — it fingerprints the TLS handshake, not
the User-Agent. The rows were pulled through a logged-in browser session by
paging the search results 25 at a time.

If we need to refresh this, the options are a headless browser (Playwright) or
the `nbainjuries` package, which reads the official NBA injury reports and
carries real filing timestamps rather than transaction dates. `nbainjuries`
needs Java 8+ (it parses PDFs via `tabula-py`), which is why it is not wired up
here yet.

## Caveat that matters for the model

These are transaction dates — when a player was placed on or activated from the
injured list — not the moment news broke. A player ruled out on the morning of
a game may appear in the log dated that day, but a same-day row has no timestamp
proving it existed before tip-off. The season replay therefore stops injury
knowledge at the previous calendar day. Interactive queries may use the user's
explicit `as_of_date`, but must describe this source as date-granular. A future
replacement should use timestamped official pre-game injury reports.
