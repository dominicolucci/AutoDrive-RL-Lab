# Lab Notes

Running observations from building this project. Newest first.

This is not a changelog — the git history covers that. It records things that
surprised me, ideas that turned out to be wrong, and results worth remembering.
Each entry follows the same shape: what I expected, what actually happened, why,
and what I changed as a result. An entry with no surprise in it does not belong
here.

**Start here if you're reading this cold:**
[The unrealistic obstacle was causing the unrealistic behaviour](#2026-09-29--the-crawling-was-the-obstacles-fault-not-the-agents) ·
[Safety scores reward standing still](#2026-08-20--a-safety-score-can-be-won-by-not-driving) ·
[The loophole was in the map, not the reward](#2026-09-28--the-loophole-was-in-the-world-not-the-reward-function) ·
[A stateful evaluator is order-dependent](#2026-09-28--the-random-baseline-was-answering-a-different-question-each-time) ·
[Imitation learning fails where the demonstrations end](#2026-08-20--the-clone-drives-like-me-and-crashes-anyway) ·
[Evaluation has to be usable outside training](#2026-08-19--evaluation-locked-inside-the-training-loop-is-half-an-evaluation)

---

## 2026-09-29 — The crawling was the obstacle's fault, not the agent's

**Expected:** removing the static obstacles was a realism tidy-up. The
interesting work was the stall rule and fault attribution; cutting the
obstacles was housekeeping on the way there.

**What happened:** it was the single largest behavioural change this project
has produced, and it fixed the headline problem on its own. The same
checkpoint, unretrained, went from **3.0 m/s over 269 m** on `dense` to
**9.9 m/s over 879 m**. The rule-based driver went from **12% safe to 80%**.
Nothing about either policy changed. Only the world did.

**Why:** an immovable block sitting in a live lane, with traffic queued behind
it, is not a driving problem — it is a wall that appears mid-motorway with no
warning and no escape once you are committed. Faced with that, crawling is
genuinely the optimal policy, and no amount of reward tuning would have said
otherwise. I had spent two sessions treating the crawling as a scoring problem
and then as a spawn-geometry problem. It was neither. It was one unrealistic
object, and the agent was responding to it correctly.

**Changed:** static obstacles are gone from every traffic preset, replaced by
vehicles genuinely travelling slower than the flow (6-9 m/s in a 29 m/s
stream). Same requirement to plan ahead and change lanes; no fiction. The
`obstacle` behaviour still exists for the lane-keeping scenario, which is what
it was built for.

**Generalised:** when an agent does something that looks stupid, check whether
the environment is asking it to. A policy responding rationally to an absurd
world is not a broken policy, and every hour spent reshaping its incentives is
an hour not spent fixing the absurdity. This is the second time on this project
the answer was in the world model rather than the reward — the first was the
[spawn geometry](#2026-09-28--the-loophole-was-in-the-world-not-the-reward-function).
I should have looked here first.

---

## 2026-09-29 — The random baseline was never "38% safe"

**Expected:** adding a stall rule would mostly affect the trained agent, since
it is the one that crawls.

**What happened:** it demolished the random baseline. Random driving now scores
**0% safe completion in every cell**, because it obstructs the lane in 93-95%
of episodes. The old figure was 29-38%.

**Why:** the old safe-completion metric asked only "did you avoid crashing and
avoid leaving the road?" A policy pressing buttons at random drifts to a halt
and sits there, satisfying both. It was never safe driving — it was a stopped
car being scored as a successful one. The stall rule does not make random
driving worse; it stops mislabelling what it always was.

**Changed:** a stall is a failure, with its own column. "Safe completion" now
means reaching the step limit without a collision, an off-road event, *or* an
obstruction.

**Worth the care it took:** the rule only counts as a stall when the lane ahead
is clear. Slowing behind traffic is ordinary driving, and a version that
punished mere slowness would have taught the agent to drive *into* stopped
queues rather than behind them. The calibration check is that the rule-based
driver never triggers it — it stalls 0% on every preset, while the trained
agent stalls 4% on `dense` and 39% on `unforgiving`. That gap is the rule
discriminating between obstruction and driving, which is exactly the job.

**Also added:** collisions are now split by fault. Being rear-ended while
driving normally is the follower's failure and costs far less than a crash the
ego caused. Without that split, the aggressive rear traffic would have punished
the agent for events it could not prevent, and it would have learned
superstitions — some unrelated behaviour that happened to correlate with fewer
unavoidable hits.

---

## 2026-09-28 — The loophole was in the world, not the reward function

> Figures below are world-model v1. The direction holds, the magnitudes moved
> once the obstacles were removed a day later.

**Expected:** the agent crawls because the reward pays too little for speed, so
the fix is to raise the speed coefficient and retrain.

**What happened:** that would have treated the symptom. Crawling isn't a bad
trade the agent is making — in this world it is genuinely, physically safe.
Every traffic car spawns at `y_m = uniform(22, sensor_range + 55)`, which is
entirely *ahead* of the ego, and `_recycle_traffic` teleported anything that
fell behind back to the front. Braking to a halt means the whole world drives
away and never comes back. An ego that does nothing but press brake for 900
steps finishes **100 out of 100** `dense` episodes untouched.

**Why:** a reward function can only price the situations the simulator can
produce. No coefficient makes stopping dangerous if nothing can ever hit you
from behind. I spent an hour reading the reward before looking at the spawn
geometry, which was the wrong end of the problem.

**Changed:** a fourth preset, `unforgiving`. Traffic also spawns behind the ego
and closes at its own cruise speed, and 40% of drivers are *inattentive* —
they act on a view of the road 0.9 s stale (`EnvConfig.reaction_delay_s`),
implemented as a fixed-length `perception_log` per car. That delay is the
danger mechanism: the safe-following model means an attentive driver can always
avoid a car that stops, so density alone changes nothing. Same brake-only
measurement on `unforgiving`: rear-ended in **23 of 100** episodes. Every
policy loses ground, and the trained agent loses it where it used to be
strongest — 100% → **88%** safe completion, so its crawling now costs 12 points
it previously got for free.

**Worth being careful about:** the three original presets had to stay *exactly*
as they were, or `BENCHMARK_current.md` would silently become a comparison
against a different simulator. Freezing them meant more than defaulting the new
fields to off — an earlier version sampled a cruise speed one call earlier
inside `_recycle_traffic`, which consumed a different number of random draws
and desynchronised every seeded world. `dense` moved from 12% to 6% for a
policy I hadn't touched. The check that caught it, and now guards it, is a
SHA-256 over every observation and reward across three presets × three seeds:
identical before and after. The regenerated table confirms it independently —
the DQN, clone and heuristic rows came back byte-for-byte.

**Generalised:** when an agent finds a degenerate strategy, ask what the world
makes possible before asking what the reward pays for.

---

## 2026-09-28 — The random baseline was answering a different question each time

**Expected:** adding a fourth cell to the benchmark would leave the other three
rows untouched, because every policy sees the same seeded worlds.

**What happened:** the `random` row moved. Not the DQN, not the clone, not the
heuristic — only random, and only in cells evaluated after the new one.

**Why:** `RandomPolicy` built one RNG in its constructor and its `reset()` did
nothing, so a single action stream ran across every episode of every cell in
order. Episode 7 of `dense` therefore depended on how many episodes had already
been drawn from that stream — which is to say, on which *other* cells were
requested. The worlds were identical, as promised. The driver was not. A
benchmark whose answer depends on the shape of the question is not a benchmark.

**Changed:** `reset()` now takes the episode seed and `RandomPolicy` reseeds
from `(base_seed, episode_seed)`, so a cell scores the same alone as it does in
company — asserted directly in `tests/test_benchmark.py`, and that test fails
if the fix is reverted. `BENCHMARK_current.md` was regenerated; only the random
rows moved (sparse 38%→29%, dense 16%→13%).

**Unexpected, and it's the interesting part:** `random` now scores *identically*
on `sparse` and `normal` — every column, to the last decimal. That is not a bug.
Given the same actions from the same start, a driver that leaves the road after
121 m never reaches any traffic, so the two worlds are indistinguishable to it.
The identical rows say out loud what the old 38%-vs-43% spread hid: the random
baseline's safety score in light traffic measures steering, not driving, and has
nothing to do with the cars.

**Generalised:** a stateful evaluator is order-dependent until proven otherwise,
and the test that catches it is cheap — score one cell twice, alone and in
company, and assert equality.

---

## 2026-08-20 — A safety score can be won by not driving

> Figures below are world-model v1 (`BENCHMARK_v1.md`). The obstacles that
> caused this were removed on 2026-09-29 and the numbers moved sharply — see
> [the crawling was the obstacle's fault](#2026-09-29--the-crawling-was-the-obstacles-fault-not-the-agents).
> The lesson stands; the measurements are historical.

**Expected:** the trained agent would look good on easy traffic and degrade on
hard traffic, with the safe-completion rate telling that story.

**What happened:** it scored **100% safe completion in every condition**, including
the hardest one — zero collisions and zero off-road events across 300 held-out
episodes. That looks like a triumph until the throughput columns are read next to
it. In dense traffic it covers **269 m at 3.0 m/s**. The same agent in sparse
traffic covers **1,522 m at 16.9 m/s**, and the rule-based driver covers 572 m
through the same dense worlds.

The clincher is the random baseline. A policy pressing buttons at random is
scored **29% safe** in sparse traffic — because it travels 121 m at 2.6 m/s and
never gets near another car.

**Why:** safe completion asks "did you avoid crashing?" and nothing else. Standing
still satisfies it perfectly. The reward function pays `0.20 × normalised speed`
for progress while collisions and off-road events cost far more, so given those
numbers, crawling is the *correct* strategy. The agent isn't failing to learn —
it learned exactly what it was told to want.

**Changed:** the benchmark now reports distance and mean speed alongside safety,
and separates collision rate from off-road rate rather than collapsing both into
one "unsafe" bucket. Evidence: `BENCHMARK_current.md`.

**Since:** the fix went into the world rather than the reward — see
[the loophole was in the world](#2026-09-28--the-loophole-was-in-the-world-not-the-reward-function).
The `unforgiving` cell makes stopping cost something, and the same agent drops
to 88% safe completion there.

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
