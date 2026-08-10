# Assessment of supplied BIN logs

The two artifacts decode without parser diagnostics, but they are not ordinary
ArduCopter flights. Both were produced by the custom `TheRocket V4.6.1`
firmware in SITL and contain a high-dynamic rocket profile.

## Inventory

| File | Size | SHA-256 | Decoded messages | Logged interval |
|---|---:|---|---:|---:|
| `00000001.BIN` | 1,441,792 B | `d1283638a591cdb7d8c4752befaa0d0d29639c429f7b66062e979de822bc512e` | 8,185 | 52.54 s |
| `00000019.BIN` | 888,832 B | `55d050c9c352657f33c046414ef0534255037fb67f2971f2357dbb18b03e5ec1` | 4,784 | 30.86 s |

## `00000001.BIN`

- Streams: 5,256 IMU, 2,627 ATT, 262 GPS, 35 MSG, 4 MODE and 1 EV record.
- GPS: all 262 samples report status 6, 10 satellites and GPS week 2412.
- IMU: two instances with 2,628 records each (about 50 Hz per instance).
- Flight stages: `INIT -> PAD_IDLE -> BOOST -> COAST -> APOGEE -> DESCENT`.
- BOOST starts at `68,614,072 us`; APOGEE is reported at `82,394,441 us`;
  parachute release is reported at `84,494,735 us` (`EV Id=51`).
- The log continues after parachute release but contains no landing, ARM or
  DISARM event.

The current generic GPS validator marks 33 samples as implausible because the
rocket reaches vertical speed above the Copter-oriented `100 m/s` limit. This is
a vehicle-profile mismatch, not evidence of file corruption.

## `00000019.BIN`

- Streams: 3,088 IMU, 1,543 ATT, 118 GPS, 31 MSG, 3 MODE and 1 EV record.
- GPS: the first sample is uninitialized (`Status=1`, `GWk=0`); the following
  117 samples report status 6, 10 satellites and GPS week 2412.
- IMU: two instances with 1,544 records each (about 50 Hz per instance).
- Flight stages: `PAD_IDLE -> BOOST -> COAST -> APOGEE -> DESCENT`.
- BOOST starts at `13,522,543 us`; APOGEE is reported at `27,903,447 us`;
  parachute release is reported at `29,802,793 us` (`EV Id=51`).
- The beginning of the pad phase is absent and there is no landing, ARM or
  DISARM event.

## Test suitability

These logs are suitable for:

- successful DataFlash decoding and schema compatibility tests;
- two-IMU selection and approximately 50 Hz sample-rate tests;
- high-dynamic acceleration, vertical-speed and altitude processing;
- GPS startup-quality filtering (`00000019.BIN`);
- custom rocket-stage and parachute-event detection;
- verifying that vehicle-specific modes and limits are not interpreted as
  ArduCopter semantics.

They are not sufficient as ground truth for:

- ArduCopter ARM/DISARM segmentation or standard Copter mode names;
- RC, GCS, battery, EKF, terrain or fence failsafe detection;
- landing, impact or crash detection;
- corrupted/truncated BIN recovery;
- large-file performance testing;
- real-aircraft behavior, because both logs identify `RC Protocol: SITL`.

The files should remain in the regression suite, but with rocket-specific
expected results and tolerances. They should complement, not replace, the SITL,
AutoTest, corrupt and anonymized real-flight corpora.
