# Limitations

This file lists, explicitly and in plain language, what this repository
**does not** demonstrate. It is written by the author of the project
to avoid any ambiguity about the scope of the work.

## 1. The visual PPO policy is not converged

`train_visual_ppo.py` is run for 20,000 timesteps on CPU. CNN-based
policies on continuous-control tasks typically require 500,000 to
1,000,000+ timesteps to converge. The trained model is not a strong
visual controller. It is an end-to-end check that the pixel
observation pipeline runs and that the policy is learning something
from pixels.

## 2. There is no formal distribution shift

The sensitivity analysis in `evaluate_observation_noise.py` adds
Gaussian noise to the observation vector at inference time. The
dynamics shift in `evaluate_dynamics_shift.py` scales mass and gravity
parameters of the simulator. The comparison in
`evaluate_domain_randomized.py` also uses simple held-out mass,
gravity, and observation-noise settings.

These are **preliminary train-test mismatch settings**, not formal
distribution shift in the ML sense (e.g. covariate shift, sim-to-real,
formal benchmark protocols, or established domain generalization
benchmarks).

The shifts are simple parametric modifications applied inside a single
MuJoCo environment. There is no formal distributional analysis, no
statistical hypothesis testing, and no comparison with established
distribution-shift benchmarks.

## 3. Domain randomization remains lightweight

The repository includes a lightweight domain-randomized PPO baseline
(`train_domain_randomized_ppo.py`) where mass and gravity are randomly
scaled during training.

This is **not** a formal sim-to-real method. It does not include
randomization over visual appearance, contact parameters, friction,
damping, actuator dynamics, sensor latency, terrain, morphology, or
real-world calibration.

The domain-randomized policy is included only to compare a simple
train-time robustness mechanism against a standard PPO policy and a
post-shift fine-tuning baseline under controlled train-test mismatch.

## 4. Adaptation is limited to a simple fine-tuning baseline

The repository includes a simple fine-tuning baseline
(`finetune_shift.py`) where the pretrained policy is further trained
for 10,000 timesteps under shifted dynamics. This recovers full
performance on the shifted setting used for fine-tuning.

This is **not** meta-learning, **not** online adaptation, **not**
test-time adaptation, and **not** any sophisticated continual learning
mechanism. It is a simple continuation of PPO training under a
different dynamics setting.

The fine-tuning baseline is included to acknowledge that policy
recovery under mismatch is possible with additional training, not to
claim a novel adaptation algorithm.

## 5. evaluate_visual_shift.py is named imprecisely

`evaluate_visual_shift.py` perturbs **state observations** based on a
severity score computed from rendered frames. It does not evaluate an
image-based policy. The state-based policy is what actually runs. The
"visual" part is only the severity score.

A more precise name would be `evaluate_visual_degradation_proxy.py`.

## 6. The CNN encoder in perception_pipeline.py is not in the loop

`perception_pipeline.py` instantiates a CNN encoder and produces
feature vectors from rendered frames, but those feature vectors are
**not** fed into the policy. The policy that controls the environment
in that script is the state-based PPO. The CNN encoder is a structural
demonstration of the visual processing component, separate from the
visual training run in `train_visual_ppo.py`.

## 7. Single seed, single environment

All results in this repository come from a single seed and a single
environment. There is no statistical analysis across seeds, no
comparison across environments, and no hyperparameter study.

The comparison between standard PPO, domain-randomized PPO, and
fine-tuned PPO should therefore be read as an exploratory controlled
experiment, not as a statistically validated benchmark result.

## 8. No baselines beyond PPO

PPO is used because it is a stable and widely used baseline available
in Stable-Baselines3 for continuous-control experiments. No comparison
is made with SAC, TD3, DDPG, model-based RL, offline RL, or imitation
learning methods.

## 9. Dynamics shifts are limited to mass and gravity

The dynamics shift evaluation scales only two parameters: body mass
and gravity. Other relevant simulator parameters, such as friction,
damping, actuator strength, contact dynamics, latency, and sensor
calibration, are not varied.

The shifts also remain within the same simulator (MuJoCo) and the same
task (`InvertedPendulum-v4`), so this is not a sim-to-real or cross-task
study.

## 10. This is not a robotics deployment project

This repository does not include ROS, real robot experiments,
manipulation tasks, 3D perception, hardware deployment, sim-to-real
transfer, or robot-system integration.

The project should be read as a small, reproducible, controlled
exploration of closed-loop policy behavior under observation corruption,
dynamics mismatch, train-time randomization, and post-shift fine-tuning.

---

The purpose of this file is to make all of the above explicit so that
reviewers, collaborators, and the author's future self can read the
repository accurately.