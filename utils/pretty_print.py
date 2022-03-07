
import pretty_errors
from pretty_errors import *
import sys

def pprint(*_args, color=None):
    if sys.stdout.isatty() and color:
        print(color, end='')
        print(*_args, end='')
        print(RESET_COLOR)
    else:
        print(*_args)

def init_pretty_error():
    pretty_errors.replace_stderr()
    pretty_errors.exception_writer.config = pretty_errors.config