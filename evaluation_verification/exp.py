"""
This script is used to run the verification experiment for a neural network model.
"""
import argparse
import sys

sys.path.append("../")

import logging
from datetime import datetime

from src import *


def main(args):

    if args.bp == "deeppoly":
        bp_method = ActRelaxationMode.DEEPPOLY
    elif args.bp == "crown":
        bp_method = ActRelaxationMode.CROWN
    elif args.bp == "rover":
        bp_method = ActRelaxationMode.ROVER_SN
    else:
        raise ValueError(f"Invalid BP method: {args.bp}")

    if args.opt == "lp":
        opt_method = OptimizationMethod.LP
    elif args.opt == "mnlp":
        opt_method = OptimizationMethod.MNLP
    else:
        opt_method = None

    log_file = None
    if args.log_name is not None:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        log_file = args.log_name
        log_file = "./logs/" + log_file + f"_{current_time}.log"

    # Set arguments for the model.
    args = Arguments(
        net_file_path=args.net_file_path,
        dataset=args.dataset,
        perturbation_radius=args.perturbation_radius,
        log_level=logging.INFO,
        act_relax_mode=bp_method,
        log_file=log_file,
        optimization_method=opt_method,
        first_sample_index=0,
        num_samples=100,
        device="cuda:0",  # noqa
        dtype="float64",
    )

    # Create the model and verify it.
    model = ModelFactory(args)
    model.prepare()
    model.build()
    model.verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment.")
    parser.add_argument("--net_file_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--perturbation_radius", type=float, default=0.01)
    parser.add_argument("--bp", type=str, default=None)
    parser.add_argument("--opt", type=str, default=None)
    parser.add_argument("--log_name", type=str, default=None)

    args = parser.parse_args()
    main(args)
