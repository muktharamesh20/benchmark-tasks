# Shake exactly 3 tablets out of a pill bottle

An uncapped amber pill bottle holding 30 tablets stands in front of the right arm. The robot must shake exactly 3
tablets out onto the table and set the bottle back down upright, without touching the tablets: the dose-counting step
of giving oral medication from a stock bottle (home health aides, O*NET 31-1121.00 task 18695).

## Files
- `shake_out_tablets.py`: the task file (scene, scoring, `success()` / `score()`).
- `partkit.py`: helper kit that builds the hollow bottle and the tablets as multi-shape rigid bodies, and sets their
  PhysX properties (CCD, per-step speed cap, friction combine mode).
- `demo.py`: the scripted demo, a standalone runner (reads simulator state to plan, as scripted demos may).
- `test_shake_out_tablets.py`: checks without the simulator (`python -m pytest -q`, with `yam_arena` importable).

## Run a trial
On a machine with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s `setup.sh`), with
`DREAM_API_KEY=<key>` in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=shake_out_tablets.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Shake exactly 3 tablets out of the bottle onto the table, then set the bottle down upright. Do not touch the tablets themselves." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/shake-out-dream

# The scripted demo, which reaches full score (DEMO_DEBUG=1 prints progress)
python demo.py shake_out_tablets.py ~/yam-eval/shake-out-demo --max-steps 1400

# One MP4 per camera from an inspect-robots run
for log in ~/yam-eval/shake-out-dream/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. The trial ends at 1.0 once exactly 3 tablets are out and the bottle has stood
upright and still on the table for 1 s; otherwise it scores `0.75·c + 0.25·u` at the last step.

## Scoring
- A tablet is **out** if its centre is outside the bottle's inside, or if it came to rest on the table outside the
  bottle at any time in the trial (tablets pushed or put back in still count).
- **c** = 1 if exactly 3 are out, 0.5 if 2 or 4, else 0.
- **u** = 0 unless at least one tablet is out; then 1 if the bottle stands upright on the table (axis within 20° of
  vertical, base within 8 mm of the table, under 1 cm/s), 0.5 if it lies still on its side (within 20° of horizontal),
  0 if it is off the table, moving, or leaning between 20° and 70°.
- Score = 0.75·c + 0.25·u.

Real trial: count the tablets outside the bottle in the final photo (table and floor), add any the video shows landing
and being put back, and check the bottle's stance from a side photo.

## Results
- Scripted demo: 1.0 (exactly 3 out, bottle upright; success at step 486).
- Dream-1: 0.0 over 2,400 steps (it lifts and sets the bottle down but never tips it far enough to pour).

## Runtime and seeds
- The scripted demo grips the bottle, tips it past level, shakes until 2 tablets are out or in the neck (one more
  follows as it turns back), tips back mouth-up to settle the count (shaking again if short), and sets the bottle down:
  about 24 s of simulated time (~490 steps at 20 Hz). On an L40S a whole run, including starting Isaac Sim, takes a
  few minutes.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. N is fixed at 3.

## Third-party assets
None: every object is built from primitives in `shake_out_tablets.py`. Bottle dimensions are from TricorBraun's
published spec for its 150 cc dark amber PET packer (38-400 neck).
