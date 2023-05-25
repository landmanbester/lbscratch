# flake8: noqa
import click


@click.group()
def cli():
    pass

from lbscratch.workers import (fledges, bsmooth,
                              restimator, gsmooth)
