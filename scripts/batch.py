import os
from subprocess import run as _run
import argparse
import glob
import sys
import pretty_errors
from pretty_errors import RED, YELLOW, GREEN, GREY, RESET_COLOR


# pretty_errors.configure(
#     separator_character = '*',
#     filename_display    = pretty_errors.FILENAME_EXTENDED,
#     line_number_first   = True,
#     display_link        = True,
#     lines_before        = 5,
#     lines_after         = 2,
#     line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
#     code_color          = '  ' + pretty_errors.default_config.line_color,
#     truncate_code       = True,
#     display_locals      = False
# )

pretty_errors.replace_stderr()
pretty_errors.exception_writer.config = pretty_errors.config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='batch.py')
    parser.add_argument('-d', '--dry-run', help='test program run', action='store_true', default=False)
    parser.add_argument('--debug', help='test program run', action='store_true', default=False)
    parser.add_argument('--dataset', help='target dataset', type=str, default='./')
    parser.add_argument('--options', type=str, default='')
    parser.add_argument('-m', '--model', type=str, action='extend', nargs='*')
    args = parser.parse_args()

    isDebug = args.debug or args.dry_run

    def pretty_print(*_args, color=None):
        if sys.stdout.isatty() and color:
            print(color, end='')
            print(*_args, end='')
            print(RESET_COLOR)
        else:
            print(*_args)

    def run(_args, **_kwargs):
        if isDebug:
            pretty_print(' '.join(_args), color=GREY)
        else:
            _run(_args, **_kwargs)

    def parse_model(_args: list):
        model = []
        for _arg in _args:
            if ',' in _arg:
                model.extend([int(a) for a in _arg.split(',')])
            if '-' in _arg and len(_arg.split('-')) == 2:
                start, end = _arg.split('-')
                model.extend([i for i in range(int(start), int(end)+1)])
            else:
                model.append(int(_arg))
        return model

    run(['python', '--version'], shell=True)
    run(['pip', '--version'], shell=True)

    SCRIPT_DIR = os.path.dirname(sys.argv[0])
    TRAIN_PROG = os.path.join(SCRIPT_DIR, 'train.py')

    options = []
    if len(args.options) > 0:
        options = args.options.split(' ')
    
    model = parse_model(args.model)

    datasets = glob.glob(os.path.join(args.dataset, '*_train.npz'))
    for train_file in datasets:
        name = train_file.replace('_train.npz', '')
        test_file = name + '_test.npz'
        stl_file = name + '.stl'

        if os.path.exists(test_file) and os.path.exists(stl_file):
            pretty_print(f'Dataset: {name}', color=GREEN)
            try:
                for i in model:
                    run(['python', TRAIN_PROG, name, '--model', str(i)] + options, shell=True)
            except Exception as e:
                pretty_errors.exception_writer.write_exception(type(e), e)
