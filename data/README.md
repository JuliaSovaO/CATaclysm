# data/: data card

This folder holds the measurements we took on a real Codey Rocky robot with a bright flashlight. They are the source of the light law and the sensor cone used in the calibrated simulation levels `C0`-`C2` (`config/env/`). The simulation itself is not stored here.

## Name and source

**CATaclysm light measurements**, collected by the team with our own Codey Rocky robot in our own test arena. The measurements were taken with `robot/measure_all.py` (mode 0, 20 samples per point). The script was uploaded to the robot from mBlock 5; the robot prints text lines to the mBlock monitor. No external dataset is used.

## Measurement conditions

| Item | Value |
|---|---|
| Robot | Makeblock Codey Rocky |
| Arena | flat floor, 200 x 200 cm |
| Room | dark (in a lit room the background reading is about 5 out of 100, which spoils the signal) |
| Light source | a bright flashlight pointed at the robot; the exact model was not recorded |
| Main sensor | light sensor in the Codey head (`codey.light_sensor.get_value()`, scale 0-100) |
| Other sensors recorded | Rocky `color_ir_sensor`: light strength (front), reflected infrared, reflected light, obstacle flag |
| Distances and angles | set by hand, so each has an error of a few centimeters and a few degrees |

## Files

| Path | What it is |
|---|---|
| `raw/light_distance_raw.txt` | text output of the distance series, as copied from the mBlock monitor; the label (distance) was added by hand after each block |
| `raw/light_angle_raw.txt` | text output of the angle series at 100 cm, same format; the label (angle) is after each block |
| `light_distance.csv` | decoded distance series, one row per point |
| `light_angle.csv` | decoded angle series, one row per point |
| `CHECKSUMS.txt` | sha256 of the four files above |

The raw files are text copied from the monitor panel; the line breaks come from the width of that panel. The CSV files are made from the raw files by joining the lines and reading the numbers.

## Schema of the CSV files

One row per measured point.

| Column | Meaning |
|---|---|
| `idx` | point number set on the robot (button B) |
| `n` | number of samples averaged (20) |
| `distance_cm` | distance from the sensor to the flashlight |
| `angle_deg` | angle between the head sensor axis and the direction to the flashlight (0 = facing the lamp) |
| `head_mean`, `head_min`, `head_max`, `head_std` | head light sensor, scale 0-100 (the sensor used for calibration) |
| `front_mean`, `front_min`, `front_max`, `front_std` | `color_ir_sensor.get_light_strength()` on Rocky; reads 100 at every distance with this flashlight, so it is not used |
| `ir_mean`, `ir_min`, `ir_max`, `ir_std` | `color_ir_sensor.get_reflected_infrared()` |
| `ref_mean`, `ref_min`, `ref_max`, `ref_std` | `color_ir_sensor.get_reflected_light()` |
| `obstacle_frac` | fraction of samples where `is_obstacle_ahead()` was true |

## What was measured

**Distance series** (robot facing the lamp, angle 0):

| Distance, cm | 10 | 20 | 30 | 50 | 90 | 130 | 150 | 175 |
|---|---|---|---|---|---|---|---|---|
| `head_mean` | 100 | 100 | 100 | 100 | 100 | 98.83 | 63.84 | 47.35 |
| `head_std` | 0 | 0 | 0 | 0 | 0 | 0.11 | 0.81 | 0.15 |

**Angle series** (distance 100 cm):

| Angle, deg | 0 | 30 | 45 | 60 | 90 |
|---|---|---|---|---|---|
| `head_mean` | 100 | 100 | 100 | 64.46 | 0.0 |


## How the data was used

| Result | Used for |
|---|---|
| Signal is about `1.44e6 / d^2` on the 0-100 scale (fit on 150 and 175 cm; fitted exponent 1.94) | `sensors.light.k` and `exponent` in `config/env/C*.yaml` |
| `cos^1.15` response to the angle (60 deg gives about 0.45 of the on-axis value, 90 deg gives 0) | `sensors.light.lobe_exponent` |
| Saturation at 100 up to about 120 cm | readings are clipped to [0, 1] in the simulation |
| Noise standard deviation 0.1 to 0.8 out of 100 | `sensors.light.sigma` |
| Front Rocky light channel is always 100 | this channel is not part of the observation |

The forward speed (15 cm/s) and the turn rate (about 150 deg/s) in the simulation come from separate motion tests (`robot/measure_all.py`, mode 2). The individual results of those tests are not stored in this folder.

## Splits

No train/test split: the data is used only to set parameters of the simulation, not to train a model.

## Personal data

None. The files contain only sensor readings of a robot and the distances and angles of the measurement.

## Scraping and external sources

Not applicable. All data was measured by the team.

## License and terms of use

Collected by the team. Free to reuse with attribution (CC BY 4.0). The license of the code in this repository is separate.

## Known limitations

- The 130 cm point lies above the pure 1/d^2 curve (98.8 against about 85), so the sensor saturates softly, not like a hard clip.
- The measurements depend on the specific flashlight and on the dark room. A different lamp or a lit room changes the readings.

## Ethical risks

Low: technical sensor data, no people, no personal information.

## Version and checksum

`CHECKSUMS.txt` records the sha256 of the files in this folder. To verify:

```bash
cd data
sha256sum -c CHECKSUMS.txt
```
