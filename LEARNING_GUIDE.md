# Learning Guide: From Highway Simulation to DQN

Use this guide with the code open. The goal is to understand why each component
exists, not merely to launch a trained model.

## 1. The simulation is the agent's world

Open `autodrive_rl/environment.py` first.

The ego car has three continuously changing physical values:

- lateral road position `ego_x_m`
- lateral speed `ego_lateral_speed_mps`
- forward speed `ego_speed_mps`

The ego car is kept at longitudinal coordinate zero. Traffic cars move relative
to it using:

```text
relative movement = (traffic speed - ego speed) × time step
```

This camera-relative trick creates a scrolling highway without storing an
unbounded map. `distance_m` still records the ego car's total forward travel.

Try this:

```python
from autodrive_rl import Action, DrivingEnv

env = DrivingEnv(scenario="lane", seed=7)
state, info = env.reset(seed=7)
next_state, reward, terminated, truncated, info = env.step(Action.ACCELERATE)
print(state)
print(reward, env.last_reward_terms)
```

## 2. Turn driving into a Markov decision process

Reinforcement learning needs five pieces:

| Piece | This project |
| --- | --- |
| State | 16 normalized vehicle and range-sensor values |
| Action | one of five discrete driving controls |
| Transition | one physics step plus traffic movement |
| Reward | progress and safe behavior minus risky behavior |
| Terminal condition | collision, off-road event, or time limit |

The state is designed to contain enough information for a useful decision. A
car directly ahead is not enough: the agent also needs adjacent-lane front and
rear gaps to decide whether a lane change is safe.

The state is not a literal road map. It is a compact occupancy description of
the drivable area around the car. That is the right first abstraction before
adding lidar grids or camera perception.

## 3. Read the reward as the specification

Find `_reward` in `environment.py`. The reward is the behavior we asked for:

```text
reward = progress
       + useful speed
       + lane centering
       - unsafe following
       - unnecessary steering
       - living cost
       - collision/off-road penalty
```

The lane-centering term is multiplied by a movement fraction. This prevents the
agent from collecting positive reward by stopping in the middle of a lane.

The agent does not understand words such as "safe" or "lane." It only discovers
which action sequences produce more total reward. A flawed reward can therefore
teach a technically optimal but unwanted behavior.

Experiments:

1. Set the progress term to zero. Does the agent learn to stop?
2. Remove the steering cost. Does it weave more often?
3. Reduce the collision penalty. Does it trade crashes for speed?
4. Increase the following penalty. Does it become overly cautious?

Change only one factor at a time and save each run's CSV metrics.

## 4. Understand Q-values

Open `autodrive_rl/dqn.py`.

The network receives one state and produces five numbers:

```text
Q(state, maintain)
Q(state, accelerate)
Q(state, brake)
Q(state, steer left)
Q(state, steer right)
```

A Q-value estimates the discounted future reward after taking that action. At
evaluation time, the policy chooses the largest value.

During training, epsilon-greedy exploration sometimes ignores the current best
action and selects randomly. Epsilon gradually falls from 1.00 to 0.05. Without
exploration, early accidental preferences can prevent the agent from discovering
better behavior.

## 5. Follow one neural-network update

`MLP.forward` applies two hidden ReLU layers and a linear output:

```text
state → affine → ReLU → affine → ReLU → five Q-values
```

This project uses the Double DQN target for one stored transition:

```text
next action = argmax(online_network(next_state))
target = reward + gamma × target_network(next_state, next action)
```

If the episode ended, the future term is removed. The error is the difference
between the online network's selected Q-value and this target.

Using one network to select the action and the slower network to evaluate it
reduces the optimistic feedback that standard DQN can develop.

`MLP.backward` applies the chain rule in reverse. `Adam.step` uses those
gradients to change every weight slightly. The implementation uses Huber loss
and gradient clipping to make large early errors less destabilizing.

## 6. Why replay memory exists

Consecutive frames are highly correlated: almost everything in frame 101 looks
like frame 100. Training only on the latest frame can make the network chase a
narrow, constantly changing sample.

`ReplayBuffer` stores many transitions. `DQNAgent.learn` randomly samples a
batch, mixing different speeds, lanes, traffic gaps, and episode outcomes. This
both reuses experience and reduces temporal correlation.

## 7. Why there are two networks

If the same rapidly changing network predicts both the current Q-value and its
own target, training can become unstable. The online network is updated every
learning step. The target network is a delayed snapshot copied periodically.

That makes the learning target temporarily steadier.

## 8. Run a controlled experiment

Start with lane keeping:

```bash
python -m autodrive_rl.train \
  --scenario lane \
  --no-curriculum \
  --episodes 100 \
  --output models/lane_only.npz \
  --metrics runs/lane_only.csv
```

Then train with the default curriculum and moving traffic:

```bash
python -m autodrive_rl.train \
  --scenario traffic \
  --episodes 300 \
  --output models/traffic.npz \
  --metrics runs/traffic.csv
```

Evaluate learning using at least:

- mean episode return
- mean distance before termination
- percentage of evaluation episodes completed safely
- collision and off-road rates
- comparison against random and heuristic policies

Do not judge only by one attractive playback. Use multiple fixed evaluation
seeds, then repeat the entire training run with multiple training seeds.

## 9. Diagnose behavior rather than guessing

| Symptom | First things to inspect |
| --- | --- |
| Agent stops | progress reward, speed reward, collision penalty balance |
| Agent leaves the road | lateral state, steering frequency, off-road penalty |
| Agent rear-ends cars | front gaps, relative speed, unsafe-following reward |
| Agent never changes lane | exploration, lane-change control cost, adjacent gaps |
| Return rises but safety falls | reward hacking; inspect every reward component |
| Training oscillates | learning rate, target update interval, replay diversity |

The important engineering habit is to turn each suspected cause into a measured
experiment.

## 10. Convert the MVP into a portfolio project

A strong second version would add:

1. A notebook or dashboard comparing three seeds and three agents.
2. Prioritized replay or a dueling head as a controlled algorithmic improvement.
3. Traffic-density difficulty levels and an evaluation matrix.
4. Sensor noise and dropped readings to test robustness.
5. Short videos showing failure cases, not only successful runs.
6. A concise report stating what improved, what did not, and why.

The portfolio story is then larger than "I made a car move." It becomes:

> I designed an RL environment, implemented DQN from first principles, created
> reproducible baselines, analyzed failure modes, and evaluated robustness under
> changing traffic conditions.
