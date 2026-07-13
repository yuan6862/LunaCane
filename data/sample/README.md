# Sample BMI270 Data

`bmi_data_walk.csv` is a small schema example for the TinyML training pipeline.
It is not large enough for real training.

Training files should be placed under `data/raw/` and named:

```text
bmi_data_<label>.csv
```

Required columns:

```text
timestamp, ax, ay, az, gx, gy, gz, acc_mag, gyro_mag, label
```

`ml/preprocess.py` uses these feature columns:

```text
ax, ay, az, gx, gy, gz, acc_mag, gyro_mag
```

Labels are read from the `label` column. Rows whose normalized label text is
`fall` are treated as fall samples; all other labels are treated as normal
motion samples.
