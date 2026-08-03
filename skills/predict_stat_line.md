---
tool: predict_stat_line
use_when: A projected points/rebounds/assists line is asked for.
---

## What it gives you

Currently nothing — `status: awaiting_input`. The regression behind it was never
built.

## Rules

- Report it as missing. Never estimate a stat line yourself from season averages and
  present it as a projection; that is exactly the invented number this whole
  interface exists to prevent.
- Projected stat lines were a stated deliverable in the proposal. Saying "not built"
  is correct and expected. Saying "LeBron will score about 25" is not.
