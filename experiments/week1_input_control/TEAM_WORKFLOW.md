# Team workflow

## Week 1 ownership

| Owner | Responsibility | Required deliverable |
| --- | --- | --- |
| Member A | environment and toolkit architecture | environment matrix, registry output, pipeline and SPipe diagram |
| Member B | system prompt and user prefix | three prompt variants, B0/B1 comparison, one success and one regression example |
| Member C | few-shot data | reviewed demonstration pool, proxy dataset, B2 comparison |
| Member D | evaluation and integration | review sheet, B3 comparison, summary table/plot, slide integration |

All members must pass the common acceptance gate in `README.md`. Ownership
means coordinating a component, not being the only person who understands it.

## Compute tiers

1. Personal environment: editing, data review, CPU tests, and tiny-model smoke
   tests.
2. Shared NTU environment: the 1.5B Week 1 ablation and backup runs.
3. Academia Sinica environment: only reviewed jobs that do not fit reliably at
   NTU, followed later by the competition models and white-box methods.

Do not share institutional accounts, SSH keys, or Hugging Face tokens. Confirm
that institutional policy and the responsible PI permit external competition
work before scheduling large runs.

## Definition of ready-to-run

A GPU experiment is ready only when:

- its code and data are present in a reviewed Git commit;
- the three input-control test files pass on CPU;
- the run request names the commit, model, config, seed, expected VRAM, and
  output location;
- a two-sample CPU or small-model integration check has completed;
- no secrets or machine-specific absolute paths are present in committed code.

## Monday presentation checklist

- environment checks from all four members;
- one AISteer360 control/pipeline/SPipe diagram;
- the B0-B3 experiment design;
- proxy data categories and separation of few-shot/dev/eval data;
- one summary table or plot;
- one steering success and one capability-regression example;
- limitations: no Starter Kit, proxy data, and non-competition model;
- next steps after the Starter Kit arrives;
- individual contribution for every team member.
