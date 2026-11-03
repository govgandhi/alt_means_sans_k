# Original calibration-control source snapshot

These source bytes match the hashes recorded in
`results/negative_sampling_control/plan.json`. The production functions have since
been corrected, so running the current diagnostic directly against that old plan
will correctly reject a source mismatch.

Reproduce the original corrected-runtime calibration control without reverting
the working tree:

```sh
python3 scripts/reproduce_calibration_control.py --output /tmp/fresh-control
```

The wrapper verifies every recorded input, copies it into a temporary experiment
root and runs the original code there. It uses the active pinned numerical
environment; follow the control protocol's build instructions first. The separate
`gensim_4_4_0_unpatched` archive retains the earlier defective-runtime sweep and is
superseded for scientific interpretation.
