import argparse
import os
import numpy as np



def args_parser():
	parser = argparse.ArgumentParser(prog="DepthEval",
									 description="evaluates depth estimation results")
	parser.add_argument("gt_folder")
	parser.add_argument("preds_folder")
	parser.add_argument("-t", "--threshold", type=float, default=1.25, help="Accuracy threshold")
	parser.add_argument("--visualize", action="store_true")

	# TODO: Model-wise postprocessing

	args = parser.parse_args()
	return args


def main():
	args = args_parser()

	gt_depth_files = sorted(os.listdir(args.gt_folder))
	pred_depth_files = sorted(os.listdir(args.preds_folder))



	print(gt_depth_files)
	print(pred_depth_files)
	print(args.visualize)
	print(args.threshold)


if __name__ == '__main__':
	main()