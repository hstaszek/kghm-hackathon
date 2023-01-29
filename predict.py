import logging.config
import os.path
from datetime import datetime
import click
import pandas as pd
from utils import utils
import joblib

logging.config.fileConfig("logging.conf")
log = logging.getLogger('base')


@click.command()
@click.option("-i", "--input-path", type=str, help="A path to input files")
@click.option("-o", "--output-path", type=str, help="A path to an output directory")
@click.option("-s", "--section", type=str, help="Section name: ZWRL_1M2C_S1_20200908_15, ZWRL_1M2C_S3_20210805_12")
def run(input_path: str, schema_path: str, output_path: str, section: str):
    
# Parse data
    # Read csv
    df = pd.read_csv(input_path)

    # Transform columns to unified format
    
    # Apply batch smoothing
        
# Load models
    #model = joblib.load(...)

# Inference
    y_pred = model.predict(X)
    log.info(f'Predictied value: {y_pred}')


if __name__ == "__main__":
    run()
