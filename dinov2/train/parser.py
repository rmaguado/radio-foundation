import argparse


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config_path", default="", help="path to config file")
    parser.add_argument(
        "--output_path",
        default="",
        type=str,
        help="Path to output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set logging level to DEBUG.",
    )
    return parser
