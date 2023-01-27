import json
import logging.config
from datetime import datetime

import click

from processing.extract import extract_fn, parse_schema
from processing.transform import transform_fn

logging.config.fileConfig("logging.conf")


@click.command()
@click.option("-c", "--config", type=str, help="A path to the processing configuration")
def run(config: str):
    with open(config, "r") as f:
        conf = json.load(f)
    common_postfix = datetime.now().strftime("%Y%m%d%H%M%S")

    schema_path = parse_schema(conf=conf)

    extract_target_path = None
    extract_conf = conf.get("process").get("extract")
    if extract_conf:
        extract_target_path = extract_fn(conf=extract_conf, postfix=common_postfix)

    transform_conf = conf.get("process").get("transform")
    if transform_conf:
        section = conf.get("section_name")
        transform_fn(
            conf=transform_conf,
            schema_path=schema_path,
            postfix=common_postfix,
            section=section,
            src_path=extract_target_path,
        )


if __name__ == "__main__":
    run()
