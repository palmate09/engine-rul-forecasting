# Data: NASA C-MAPSS Turbofan Engine Degradation Simulation

## Source

- NASA Prognostics Center of Excellence (PCoE) Data Set Repository
- Dataset: **Turbofan Engine Degradation Simulation Data Set**
- Download: `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`
- NASA Open Data landing page: `https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data`

## Citation

> A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
> NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

Related paper:

> A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for
> Aircraft Engine Run-to-Failure Simulation," PHM 2008.

## Files used

| File | Meaning |
|---|---|
| `train_FD001.txt` | 100 run-to-failure engines, 1 operating condition, 1 fault mode |
| `test_FD001.txt` | 100 truncated trajectories |
| `RUL_FD001.txt` | One true RUL value per test engine |

Each data row has 26 space-separated columns:

1. unit number
2. cycle
3–5. operational settings 1–3
6–26. sensor measurements 1–21

## Important limitations

- The data are **simulated** by the C-MAPSS turbofan simulator, not field data from a real fleet.
- FD001 has one operating condition and one fault mode; FD002/FD004 are harder.
- No maintenance history, cost data, or failure-mode labels per test engine are provided.
- The NASA page does not state a standard open license. Cite the source and do not present
  the data as your own. Check the repository terms before redistributing the raw files.

## Download

From the project root:

```bash
python scripts/download_data.py
```

The script verifies that all 12 train/test/RUL files are present.
