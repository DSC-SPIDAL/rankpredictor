5.DeployModel
====================

1. pit stop and crash analysis

To use $C in the stream of the log records to indentify pitstop is feasible.

Refer to [pitstop analysis](pitstop_crash_data_analysis.md)

But it is hard to tell the differences between data loss and the aftermath of a crash.

```
#combine the $C and $P for all the cars
python -m indycar.rplog_stream.py --extract --input indycar-2018.log --output indy2018 --telemetry --combine_c

```

2. model deployment 

demo in [predictor](predictor/)
