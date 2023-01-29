## Description
Solution for CuValley Hack 2023.
### How to run:
1. Get Python 3.9
2. Run `pip -r requirements.txt`

```
Example usage to process dataset and train models (dataset is unpacked to data):
process.py \
    -i 'data/ZWRL_1M2C_S1_20200908_15/*' \
    -x data/ZWRL_1M2C_S1_20200908_15/His_2c_zmienne_s1.xlsx \
    -o output \
    -s ZWRL_1M2C_S1_20200908_15
```

### Notes
File `LMAM_HC212B_NDFT01---_TFI` has incorrect headers, it has to be corrected before running the script.