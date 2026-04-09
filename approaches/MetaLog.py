import sys

sys.path.extend([".", ".."])

from approaches.supervised_protocol import (
    MetaLog,
    build_arg_parser,
    mamba_conv,
    mamba_expand,
    mamba_state,
    mamba_variant,
    lstm_hiddens,
    num_layer,
    run_direction,
)


def main():
    argparser = build_arg_parser()
    args, _ = argparser.parse_known_args()
    run_direction('hdfs_to_bgl', args)


if __name__ == '__main__':
    main()
