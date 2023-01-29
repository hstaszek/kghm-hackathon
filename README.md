## Description
Solution for CuValley Hack 2023.
### How to run:
1. Get Python 3.9
2. Run `pip -r requirements.txt`
3. Run `process.py`

Parameter description:
```
"-i" - "A path to input files"
"-x" - "A path to a schema file in xlsx"
"-o" - "A path to an output directory"
"-s" - "Section name, f.e. ZWRL_1M2C_S1_20200908_15, ZWRL_1M2C_S3_20210805_12"
```
Example usage to process dataset and train models (dataset is unpacked to `data`):
```
process.py \
    -i 'data/ZWRL_1M2C_S1_20200908_15/*' \
    -x data/ZWRL_1M2C_S1_20200908_15/His_2c_zmienne_s1.xlsx \
    -o output \
    -s ZWRL_1M2C_S1_20200908_15
```

### Notes
* `exploration` contains our notebooks we used during exploration phase.  
* File `LMAM_HC212B_NDFT01---_TFI` has incorrect headers, it has to be corrected before running the script.