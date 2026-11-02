### Synthetic staggered DID (true effect = 0.5)

Sample - observations: 24000, firms: 2000, years: 2010-2021, treated_share: 0.281

| Method | Coef. | Std. err. | p-value | 95% CI |
|---|---:|---:|---:|---|
| DML - Random Forest | 0.5880*** | 0.0269 | 0.000 | [0.5352, 0.6407] |
| DML - Neural Network | 0.4723*** | 0.0264 | 0.000 | [0.4205, 0.5241] |
| DML - Gradient Boosting | 0.4998*** | 0.0265 | 0.000 | [0.4478, 0.5517] |
| DML - Elastic Net | 0.5001*** | 0.0271 | 0.000 | [0.4470, 0.5531] |
| DML - Ridge | 0.5000*** | 0.0270 | 0.000 | [0.4470, 0.5530] |
| TWFE OLS (benchmark) | 0.4991*** | 0.0271 | 0.000 | [0.4461, 0.5521] |

*** p<0.01, ** p<0.05, * p<0.1
