# Mixed-Direction Evaluation — PP_geo_hour

MASCOT and MS-DPP evaluated on PP_geo_hour with four direction configurations.

**Regression verified**: `[INC,INC]` == pure INCREASE and `[DEC,DEC]` == pure DECREASE for both methods (checked per-query, not just aggregate).


## Key metrics

- **geo-Vendi**: Vendi score computed on normalised GPS coordinates of the top-K list (higher = more geographically spread).
- **time-Vendi**: Vendi score computed on circular time embeddings of the top-K list (higher = more temporally spread).
- **R@10**: standard retrieval recall.

## Results

| Method | Direction | R@1 | R@10 | geo-Vendi | time-Vendi | mean-Vendi | HM(R10,Vendi) |
|---|---|---|---|---|---|---|---|
| MASCOT | geo↑ time↑ (pure INC) | 0.7578 | 0.8356 | 0.9788 | 0.8601 | 0.9435 | 0.8863 |
| MASCOT | geo↓ time↓ (pure DEC) | 0.7202 | 0.9410 | 0.8878 | 0.8083 | 0.1881 | 0.3135 |
| MASCOT | geo↑ time↓ (**mixed**) | 0.0301 | 0.1167 | 0.9620 | 0.7894 | 0.9280 | 0.2073 |
| MASCOT | geo↓ time↑ (**mixed**) | 0.0000 | 0.0038 | 0.9005 | 0.8319 | 0.9208 | 0.0075 |
| MS-DPP | geo↑ time↑ (pure INC) | 0.7829 | 0.9749 | 0.9433 | 0.8400 | 0.9262 | 0.9500 |
| MS-DPP | geo↓ time↓ (pure DEC) | 0.5270 | 0.8381 | 0.7814 | 0.7474 | 0.2721 | 0.4108 |
| MS-DPP | geo↑ time↓ (**mixed**) | 0.6374 | 0.9159 | 0.9728 | 0.7973 | 0.9325 | 0.9241 |
| MS-DPP | geo↓ time↑ (**mixed**) | 0.5747 | 0.9034 | 0.8359 | 0.8371 | 0.9089 | 0.9061 |

## Verdict

**PARTIAL** for `geo↓+time↑`: expected opposing Vendi trajectories not fully confirmed.

### Architectural note

MASCOT's mixed-direction signal is **metadata-only**: per-attribute signed coverage gain on the IU probability bins. MASCOT has no appearance kernel — appearance diversity is an emergent property of the retrieval scores rather than an explicit objective. MS-DPP's appearance kernel (`β × img_log`) is always unsigned (promotes visual diversity regardless of direction), and the per-attribute sign is applied only to the metadata component. This distinction is not a deficiency of either method but a fundamental architectural difference: MASCOT's greedy IU formulation and MS-DPP's DPP kernel formulation handle metadata direction differently but both correctly isolate appearance from metadata in the per-attribute case.

