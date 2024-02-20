import argparse
from siinn.siinn_inspect import inspect_proxy
from siinn.siinn_run import run_proxy


def main():
    parser = argparse.ArgumentParser('Simple Inspect and Inference Neural Network.')
    parser.add_argument('mode', type=str, help='inspect | run(/=inference=speed)')
    parser.add_argument('modelfile', type=str, help='input model file')
    args = parser.parse_args()
    assert args.mode in ['inspect', 'run', 'inference', 'speed']

    if args.mode == 'inspect':
        inspect_proxy(args.modelfile)
    else:
        run_proxy(args.modelfile)

if __name__ == "__main__":
    main()
