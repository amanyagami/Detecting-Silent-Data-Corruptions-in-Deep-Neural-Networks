Clean accuracy 93.35%; 400 random faults x 8 images = 3200 trials (masked 3060, sdc 139, nonfinite 1).

| detector | FPR@0.01 (clean) | TPR on SDC | AUROC clean vs SDC |
|---|---|---|---|
| tau1 | 0.003 (6/1867) | 0.849 [0.780, 0.899] (n=139) | 0.951 |
| tau2 | 0.009 (16/1867) | 0.906 [0.847, 0.945] (n=139) | 0.966 |
| tau3 | 0.000 (0/1867) | 0.000 [0.000, 0.027] (n=139) | 0.619 |
| tau1+2 | 0.003 (6/1867) | 0.849 [0.780, 0.899] (n=139) | 0.952 |
| tau1/k+2 | 0.005 (9/1867) | 0.878 [0.813, 0.922] (n=139) | 0.961 |
| all | 0.003 (6/1867) | 0.849 [0.780, 0.899] (n=139) | 0.953 |

SDC detection (all components, FPR 0.01) by flipped field:

| field | SDC trials | detected | rate |
|---|---|---|---|
| sign | 0 | 0 | nan |
| exponent | 139 | 118 | 0.849 |
| mantissa-high | 0 | 0 | nan |
| mantissa-low | 0 | 0 | nan |

Masked flips flagged: 28/3060 (harmless to the output).
