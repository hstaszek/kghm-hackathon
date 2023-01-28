import logging.config
import os.path
from datetime import datetime

import click

from processing.extract import parse_schema, collect_fn, transform_fn

logging.config.fileConfig("logging.conf")
log = logging.getLogger('base')


@click.command()
@click.option("-i", "--input-path", type=str, help="A path to input files")
@click.option("-x", "--schema-path", type=str, help="A path to a schema file in xlsx")
@click.option("-o", "--output-path", type=str, help="A path to an output directory")
@click.option("-s", "--section", type=str, help="Section name: ZWRL_1M2C_S1_20200908_15, ZWRL_1M2C_S3_20210805_12")
def run(input_path: str, schema_path: str, output_path: str, section: str):
    """
    Example usage:
    main.py \
        -i data/ZWRL_1M2C_S1_20200908_15/* \
        -x data/ZWRL_1M2C_S1_20200908_15/His_2c_zmienne_s1.xlsx \
        -o output \
        -s ZWRL_1M2C_S1_20200908_15
    """
    common_postfix = datetime.now().strftime("%Y%m%d%H%M%S")

    output_dir = os.path.join(output_path, section, common_postfix)
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"created target dir: {os.path.dirname(output_path)}")

    schema_path = parse_schema(schema_path)

    extract_target_path = collect_fn(input_path, output_dir)
    load_target_path = transform_fn(extract_target_path, schema_path, output_dir)

    # find_corelations(
    #     src_csv=PATH_TO_COMBO_.CSV_FILE,
    #     schema_path = os.path.join(BASEDIR, DATASET, SCHEMA),
    #     target_path = target_dir
    # )


if __name__ == "__main__":
    run()
