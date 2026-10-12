# MultiPosture error analysis

Best exported track: `HistGradientBoosting` + `hybrid`. The following observations use only the participant-held-out final test subjects: `6, 8`.

## Final test confusion matrix

Rows are true labels and columns are predicted labels in this order: TLB, TLF, TLL, TLR, TUP.

```text
      TLB   TLF   TLL   TLR   TUP
  TLB    11     0     0     0     0
  TLF     0   427     0     0     0
  TLL     0     0    60     0     0
  TLR     0     0     0    60     0
  TUP     0     0     0     0   192
```

## Most frequent confusions

- No off-diagonal test errors were observed in this holdout.

## Test-subject summary

- Subject `6`: n=480, accuracy=1.0000, macro F1=1.0000
- Subject `8`: n=270, accuracy=1.0000, macro F1=1.0000

This analysis is limited by 13 total subjects and a two-subject final test holdout.
