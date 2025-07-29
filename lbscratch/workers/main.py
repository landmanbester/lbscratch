# flake8: noqa
import click
from importlib.metadata import version


@click.group()
@click.version_option(version=version("lbscratch"), prog_name="lbs")
def cli():
    pass


from lbscratch.workers import (
    fledges,
    bsmooth,
    restimator,
    gsmooth,
    flagger,
    chanflags,
    clip_gain_amps,
)
