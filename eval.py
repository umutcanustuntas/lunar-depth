import argparse
import os
import numpy as np

from metrics import evaluate
from PIL import Image

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


	for pred_file, gt_file in zip(gt_depth_files, pred_depth_files):
		pred_file = os.path.join(args.preds_folder, pred_file)
		gt_file = os.path.join(args.gt_folder, gt_file)
		pred_image = Image.open(pred_file)
		gt_image = Image.open(gt_file)


		pred_arr = np.array(pred_image)
		gt_arr = np.array(gt_image)
#		gt_arr = gt_arr + np.ones(gt_arr.shape)

		evaluate(gt=gt_arr, pred=pred_arr)

		break
if __name__ == '__main__':
	main()