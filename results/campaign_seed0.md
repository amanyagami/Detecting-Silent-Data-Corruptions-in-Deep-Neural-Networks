Clean accuracy 92.90%; 300 random faults x 16 images = 4800 trials (masked 4761, sdc 39, nonfinite 0).

| detector | FPR@0.01 (clean) | TPR on SDC | AUROC clean vs SDC |
|---|---|---|---|
| tau1 | 0.004 (7/1858) | 0.897 [0.764, 0.959] (n=39) | 0.988 |
| tau2 | 0.008 (15/1858) | 0.974 [0.868, 0.995] (n=39) | 0.992 |
| tau3 | 0.000 (0/1858) | 0.000 [0.000, 0.090] (n=39) | 0.419 |
| tau1+2 | 0.004 (7/1858) | 0.897 [0.764, 0.959] (n=39) | 0.988 |
| tau1/k+2 | 0.004 (8/1858) | 0.974 [0.868, 0.995] (n=39) | 0.992 |
| all | 0.004 (7/1858) | 0.897 [0.764, 0.959] (n=39) | 0.988 |

SDC detection (all components, FPR 0.01) by flipped field:

| field | SDC trials | detected | rate |
|---|---|---|---|
| sign | 0 | 0 | nan |
| exponent | 39 | 35 | 0.897 |
| mantissa-high | 0 | 0 | nan |
| mantissa-low | 0 | 0 | nan |

Masked flips flagged: 21/4761 (harmless to the output).
