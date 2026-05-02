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
parameters of the simulator. These are **preliminary train-test
mismatch settings**, not formal distribution shift in the ML sense
(e.g. covariate shift, sim-to-real, domain randomization, formal
benchmark protocols).

The shifts are simple parametric modifications applied at evaluation
time. There is no formal distributional analysis, no statistical
hypothesis testing, and no comparison with established
distribution-shift benchmarks.

## 3. Adaptation is limited to a simple fine-tuning baseline

The repository includes a simple fine-tuning baseline
(`finetune_shift.py`) where the pretrained policy is further trained
for 10,000 timesteps under shifted dynamics. This recovers full
performance on the shifted setting.

This is **not** meta-learning, **not** online adaptation, **not**
test-time adaptation, and **not** any sophisticated continual learning
mechanism. It is a simple supervised continuation of training under a
different dynamics setting.

The fine-tuning baseline is included to acknowledge that policy
recovery under mismatch is possible with additional training, not to
claim a novel adaptation algorithm.

## 4. evaluate_visual_shift.py is named imprecisely

`evaluate_visual_shift.py` perturbs **state observations** based on a
severity score computed from rendered frames. It does not evaluate an
image-based policy. The state-based policy is what actually runs. The
"visual" part is only the severity score.

## 5. The CNN encoder in perception_pipeline.py is not in the loop

`perception_pipeline.py` instantiates a CNN encoder and produces
feature vectors from rendered frames, but those feature vectors are
**not** fed into the policy. The policy that controls the environment
in that script is the state-based PPO. The CNN encoder is a structural
demonstration of the visual processing component, separate from the
visual training run in `train_visual_ppo.py`.

## 6. Single seed, single environment

All results in this repository come from a single seed and a single
environment. There is no statistical analysis across seeds, no
comparison across environments, and no hyperparameter study.

## 7. No baselines beyond PPO

PPO is used because it is the default Stable-Baselines3 algorithm for
continuous control. No comparison is made with SAC, TD3, or other
algorithms.

## 8. Dynamics shifts are limited to mass and gravity

The dynamics shift evaluation scales only two parameters: body mass
and gravity. Other relevant simulator parameters (friction, damping,
actuator strength, contact dynamics) are not varied. The shifts also
remain within the same simulator (MuJoCo) and the same task
(InvertedPendulum), so this is not a sim-to-real or cross-task study.

---

The purpose of this file is to make all of the above explicit so that
reviewers, collaborators, and the author's future self can read the
repository accurately.