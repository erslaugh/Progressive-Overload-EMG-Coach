# EMG + IMU Workout Analysis Dashboard

This is the local desktop version of the dashboard.

## Inputs
- Name
- Weight and units
- Exercise name
- Backyard Brains / Spike Recorder `.wav` EMG file
- `.zip` containing Phyphox accelerometer and gyroscope CSV exports

The EMG and IMU recordings do **not** have to begin at the same time.

## What the pipeline currently does
1. Loads and band-pass filters EMG.
2. Calculates a rolling EMG RMS envelope.
3. Extracts Phyphox gyroscope and accelerometer files from the ZIP.
4. Finds the dominant gyroscope axis.
5. Detects repetitions from gyroscope movement peaks.
6. Calculates per-rep angular-speed and acceleration-jerk features.
7. Estimates an EMG ↔ IMU timing offset from the first clear activity onset.
8. Assigns EMG RMS features to each detected rep.
9. Measures within-set velocity drop, EMG RMS rise, and jerk rise.
10. Produces one of:
   - INCREASE WEIGHT
   - DO MORE REPS / KEEP WEIGHT
   - KEEP WEIGHT
   - STOP / END SET
11. Saves per-rep features to CSV and a session summary to JSON.

## Install
Open Command Prompt or PowerShell in this folder:

    py -m pip install -r requirements.txt

## Run

    py workout_dashboard.py

## Important calibration note
The current recommendation thresholds are intentionally marked **PROVISIONAL**.
They give you a working end-to-end prototype, but should not be treated as validated
exercise-prescription thresholds.

For the hackathon data collection, save:
- subject ID
- weight
- exercise
- set number
- completed reps
- whether the subject intentionally stopped / reached failure
- perceived exertion or reps-in-reserve
- whether form was judged acceptable
- EMG WAV
- IMU ZIP
- a synchronization event when possible

Those labels are what we can use next to calibrate the recommendation logic.


## Most recent data mode
The dashboard now defaults to your Downloads folder. Choose a different Data Folder if needed.
Every time you press **Run Analysis**, it automatically selects the newest `.wav` and newest `.zip`
found inside that folder (including subfolders). The file fields remain visible so you can verify
which recordings will be analyzed or manually override them.
