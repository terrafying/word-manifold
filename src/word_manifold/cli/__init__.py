"""
Command line interface for word-manifold.
"""

import click
from .commands.magic import magic

@click.group()
def cli():
    """Word Manifold CLI"""
    pass

cli.add_command(magic)

if __name__ == '__main__':
    cli() 