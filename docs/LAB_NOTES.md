# Lab Notes

Running observations from building this project. Newest first.

This is not a changelog — the git history covers that. It records things that
surprised me, ideas that turned out to be wrong, and results worth remembering.
Each entry follows the same shape: what I expected, what actually happened, why,
and what I changed as a result. An entry with no surprise in it does not belong
here.

**Start here if you're reading this cold:**
[Safety scores reward standing still](#2026-08-20--a-safety-score-can-be-won-by-not-driving) ·
[Imitation learning fails where the demonstrations end](#2026-08-20--the-clone-drives-like-me-and-crashes-anyway) ·
[Evaluation has to be usable outside training](#2026-08-19--evaluation-locked-inside-the-training-loop-is-half-an-evaluation)

---

## 2026-08-20 — A safety score can be won by not driving

**Expected:** the trained agent would look good on easy traffic and degrade on
hard traffic, with the safe-completion rate telling that story.

**What happened:** it scored **100% safe completion in every condition**, including
the hardest one — zero collisions and zero off-road events across 300 held-out
episodes. That looks like a triumph until the throughput columns are read next to
it. In dense traffic it covers **269 m at 3.0 m/s**. The same agent in sparse
traffic covers **1,522 m at 16.9 m/s**, and the rule-based driver covers 572 m
through the same dense worlds.

The clincher is the random baseline. A policy pressing buttons at random is
scored **38% safe** in sparse traffic — because it travels 115 m at 2.4 m/s and
never gets near another car.

**Why:** safe completion asks "did you avoid crashing?" and nothing else. Standing
still satisfies it perfectly. The reward function pays `0.20 × normalised speed`
for progress while collisions and off-road events cost far more, so given those
numbers, crawling is the *correct* strategy. The agent isn't failing to learn —
it learned exactly what it was told to want.

**Changed:** the benchmark now reports distance and mean speed alongside safety,
and separates collision rate from off-road rate rather than collapsing both into
one "unsafe" bucket. Evidence: `BENCHMARK_current.md`.

**Still open:** is this a property of the training method or of one lucky run?
Three seeds (`run_s1/2/3`) exist and have not been benchmarked. And the causal
test — raising the speed coefficient and retraining — hasn't been run yet.

---

## 2026-08-20 — The clone drives like me, and crashes anyway

**Expected:** cloning my own driving would produce a mediocre but basically
functional driver. I don't crash much, so a policy imitating me shouldn't either.

**What happened:** it collides in **94% of sparse episodes and 97% of dense
episodes** — while driving at **22.6 m/s**, faster than the RL agent's 16.9. It
copied my speed and my confidence and none of my safety.

**Why:** it only ever saw states I actually visited. I never demonstrated
recovering from a near-collision, because I never got into one. So the moment the
policy drifts even slightly off the trajectory I'd have taken, it's in a
situation with no training signal at all, and small errors compound into a
crash. This is covariate shift, and it's the standard failure of behaviour
cloning rather than a bug in the implementation.

**Changed:** nothing yet — the result is the point. It's the cleanest
demonstration in the project of *why* reinforcement learning exists: the RL agent
explores bad states during training and learns what to do in them; the clone
never sees one until it's too late.

**Still open:** the clone does better on `normal` (24% safe) than on `sparse` (6%),
which is backwards. Best guess is speed — it drives 6 m/s faster in sparse
traffic, so its errors have less time to be corrected. Unverified.

**Worth trying:** DAgger. Let the clone drive, correct it where it goes wrong, add
those corrections to the training set, repeat. That directly attacks the gap this
result exposes.

---

## 2026-08-20 — Never assume a UI change is visually neutral

**Expected:** restyling the launcher was cosmetic, so tests passing meant it was fine.

**What happened:** three separate visual bugs, all invisible to the test suite.
Panels fired their change callbacks *during* construction, before the widget they
updated existed. The `clam` theme lifts a selected notebook tab by default, which
clipped it against the frame edge. And `clam` draws checkbox ticks with
`indicatorforeground`, ignoring the `indicatorcolor` option other themes use — so
every setting I wrote for the checked state landed on a property nothing reads,
and checked boxes looked identical to unchecked ones.

**Why:** a passing test proves the code ran, not that the result looks right.
Nothing in a unit test can see a clipped tab.

**Changed:** render and look. Screenshotting the real window under a virtual
display caught all three in minutes, and also proved that an artifact I'd assumed
I introduced — a connector line painting over the step labels — was there before I
touched anything.

---

## 2026-08-19 — Evaluation locked inside the training loop is half an evaluation

**Expected:** per-difficulty evaluation already existed, since training logs
`eval_sparse/normal/dense` columns to CSV.

**What happened:** it could only ever score the agent it was *currently training*.
There was no way to take a saved checkpoint — the behaviour clone, an older run,
a seed from last week — and score it. Which meant the one comparison the project
most needed, imitation versus reinforcement, was impossible to make.

**Why:** the evaluation function took a live agent object rather than something
loadable, and lived inside `train.py` where nothing else could call it.

**Changed:** `autodrive_rl/benchmark.py`, a standalone command that scores any
policy — checkpoint, rule-based, or random — on identical held-out worlds. Two
consequences worth noting. Every policy must see the *same seeds*, or the table
measures luck instead of driving; that property is now the first thing the tests
check. And the markdown output prints the command that regenerates it, because a
results table nobody can reproduce is a screenshot with extra steps.

---

## 2026-08-19 — A benchmark nobody can regenerate quietly rots

**Expected:** `BENCHMARK.md` documented current performance.

**What happened:** its numbers had been measured before the traffic model gained
physics-based following, speed limits, gap-aware merging and obstacles. They
described a simulation this repository no longer contains. Nothing on the page
said so, and there was no command anywhere that would reproduce the four-row table
it showed.

**Why:** the results were assembled by hand from a training log. Anything
assembled by hand gets recreated by hand, which means never.

**Changed:** the old table now opens by stating plainly that it is historical, and
names the command that replaces it. Lesson generalised: a results file should
carry the instructions for regenerating itself, or it will be wrong within weeks
and no one will notice.

---

## 2026-07-23 — A car changing lanes occupies both lanes

**Expected:** preventing traffic cars from overlapping was a matter of checking
gaps in the lane each car is in.

**What happened:** cars still ended up in the same physical space. Two drivers
would decide to merge into the same gap at the same moment, each seeing it as
empty because the other was still nominally in its old lane.

**Why:** a lane change is an animation, not an instant. For its duration the car
is physically present in both lanes, but the gap checks treated it as belonging
to exactly one.

**Changed:** `_occupied_lanes` reports both lanes for a car mid-change, and every
gap check — following, merging, spawning, obstacle placement — uses it. Lane
changes now abort and reverse smoothly if the target gap collapses, with the
less-committed driver yielding; past 50% progress the change is committed and
others yield instead. Pinned by
`test_no_two_cars_ever_overlap_geometrically`, which asserts a rectangle-overlap
invariant over full rollouts rather than checking a handful of cases.

**Generalised:** when something is animating between two states, it is usually in
both. Modelling it as being in exactly one is where the bug hides.

---

## 2026-07-19 — Put a seam in front of the tool you don't control

**Expected:** adding MLflow experiment tracking meant importing MLflow and calling
it from the training loop.

**What happened:** that would have made every test require MLflow, made offline
runs awkward, and welded the training code to one vendor's API.

**Why:** the training loop doesn't care *where* metrics go. It cares that
something receives them.

**Changed:** a project-owned tracker interface with a real MLflow implementation
and a no-op twin. Tests use the twin and never touch MLflow; `--no-tracking` runs
offline; swapping tools later means writing one class, not editing the training
loop. (MLflow is pinned below 3.0 — 3.x deprecates the local file store this uses.)

---

## Open questions

Things I've noticed and can't yet explain.

- The clone scores **worse on sparse (6%) than on normal (24%)** traffic. Backwards.
  Speed is the leading suspect, unconfirmed.
- Does the crawling behaviour reproduce across training seeds, or was one run
  unlucky? Three checkpoints exist, unbenchmarked.
- Which part of the `dense` preset actually breaks each policy? It changes three
  things at once — 9→14 cars, 0→2 obstacles, 0%→50% lane-changers — so current
  results cannot attribute the failure. My guess is the obstacles, because they're
  static and the curriculum never introduces them.
- Is 99% meaningfully better than 89% at n=100? The design is paired (identical
  seeds per policy), so McNemar's test would answer it properly.

---

## Template

```markdown
## YYYY-MM-DD — Short claim, not a topic

**Expected:**

**What happened:**

**Why:**

**Changed:**
```

Write the entry when the surprise is fresh. Delete it if, on rereading, nothing
in it was surprising.
