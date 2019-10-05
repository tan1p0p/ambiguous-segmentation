import argparse

def argment_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--no_write', action='store_true', \
                        help='Run without making any output files.')
    return parser.parse_args()
