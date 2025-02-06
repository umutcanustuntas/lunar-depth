import torch
import numpy as np


def abs_rel(gt, pred):
	diff = np.abs(pred - gt) / gt

	return diff.mean()

def sq_rel(gt, pred):
	diff = np.square(pred - gt) / gt

	return diff.mean()

def rmse(gt, pred):
	diff = np.square(pred - gt).mean()

	return np.sqrt(diff)

def rmse_log(gt, pred):
	log_diff = np.square(np.log(pred) - np.log(gt)).mean()

	return np.sqrt(rmse_log)

def accuracy(gt, pred, thres=1.25):
	delta = np.maximum(pred/gt, gt/pred)
	valid = delta < thres
	return valid.mean()

def evaluate(gt, pred):

	abs_rel_score = abs_rel(gt, pred)
	sq_rel_score = sq_rel(gt, pred)
	rmse_score = rmse(gt, pred)
	rmse_log_score = rmse_log(gt, pred)

	acc_1 = accuracy(gt, pred, 1.25)
	acc_2 = accuracy(gt, pred, 1.25**2)
	acc_3 = accuracy(gt, pred, 1.25**3)

	print(f"abs_rel_score: {abs_rel_score}")
	print(f"sq_rel_score: {sq_rel_score}")
	print(f"rmse_score: {rmse_score}")
	print(f"rmse_log_score: {rmse_log_score}")
	print(f"acc_1: {acc_1}")
	print(f"acc_2: {acc_2}")
	print(f"acc_3: {acc_3}")

def main():
	print("Testing...")

if __name__ == '__main__':
	main()