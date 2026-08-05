# Adaptive bond-dimension trajectories

`lomps-run-adaptive` starts an actual trajectory at a modest bond dimension and
promotes it only when the current manifold cannot accept the next update. This
is intended for product and other weakly entangled starts where directly
embedding into a very large bond dimension can make the first optimization
poorly conditioned.

At a promotion from `D_old` to `D_new`, the last accepted `D_old` tensor remains
the exact source of the next evolved target. `lomps-run` lifts that tensor only
to initialize the `D_new` optimizer. A promotion therefore changes the
variational manifold without redefining the physical update being fitted.

Example for the standard Y+ quench at `L=6`:

```bash
lomps-run-adaptive \
  --initial-A data/y_plus.npy \
  --output-dir runs/nonintegrable_y_plus_L6_adaptive \
  --block-length 6 \
  --bond-dimensions 4,6,8,12,16,20,24,28,32,36,40,42 \
  --steps 20000 \
  --delta-t 1e-3 \
  --initial-first-step-accept-cost 1e-15 \
  --handoff-accept-cost 1e-14 \
  --accept-cost 1e-14 \
  --embedding-noise-amplitude 1e-6 \
  --perturb-amplitudes 0.1,0.3,0.6,1.0 \
  --perturbations-per-amplitude 1
```

The command does not alter `lomps-run`. Each ordinary child run is stored in a
separate `segment_NNN_DD` directory. `manifest.json` records the status,
accepted time range, failure that triggered each promotion, and handoff paths.
The `handoffs` directory contains both the accepted tensor and its right fixed
point. No failed optimizer candidate is used as a source for a later segment.

In a promoted segment, state zero is the lifted optimizer seed. For a stitched
physical trajectory, retain the final accepted state of the preceding segment
at the handoff time and begin the promoted segment at its first accepted update.
The manifest contains the paths and times needed to do this without ambiguity.

Promotion is deliberately conservative: the child first exhausts its configured
LM retry and perturbation seeds. Only `failed_fixed_target_multistart` advances
to the next bond dimension. Pauses, interrupts, and process errors stop the
adaptive controller instead of being interpreted as insufficient capacity.

The first product fit and later promotions have separate thresholds. The former
defaults to `1e-15`; handoffs default to the recurrent `1e-14` threshold because
they combine a dimension change with an ordinary physical time update.

The first version deliberately refuses an existing top-level output directory;
it does not yet resume the adaptive controller itself. Every child segment is a
normal checkpointed `lomps-run`, so an interrupted segment remains inspectable
and individually resumable. From a source checkout where the console entry
point has not been refreshed, the equivalent invocation is
`python -m lomps.adaptive`.
