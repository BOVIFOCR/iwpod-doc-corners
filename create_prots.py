from glob import glob
import random
import json

import argparse
import os

def get_files(face="both", fdir="rg_dataset/"):
	if face == "both":
		ret = [x.split("/")[-1] for x in glob(f"{fdir}wild/*") if not x.endswith(".txt")]
	elif face == "front":
		with open(f"{fdir}front_files.txt", "r") as fd:
			ret = [x.strip().split(" ")[0] for x in fd.readlines()]
	elif face == "back":
		with open(f"{fdir}back_files.txt", "r") as fd:
			ret = [x.strip().split(" ")[0] for x in fd.readlines()]

	return ret

if __name__ == "__main__":
	ps = argparse.ArgumentParser()
	ps.add_argument("--nfold", "-n", type=int, default=10)
	ps.add_argument("--protocol", "-p", type=str, default="closeup")
	ps.add_argument("--face", "-f", type=str, default="both")
	ps.add_argument("--dataset_dir", "-d", type=str, default="rg_dataset/")
	ps.add_argument("--output_dir", "-o", type=str, default="prots/")
	args = ps.parse_args()

	part = args.protocol
	n = args.nfold
	face = args.face
	dataset_dir = args.dataset_dir
	output_dir = args.output_dir

	if not os.path.exists("rg_dataset/prots2/"):
		os.mkdir("rg_dataset/prots2/")

	fs = get_files(face=face, fdir=dataset_dir)
	random.shuffle(fs)

	# Split the list of all files into n bins
	def split_files(files, n):
		ret = []
		for i in range(n):
			ret.append([])
		i = 0
		for f in files:
			ret[i%n].append(f)
			i += 1
		return ret

	rr = split_files(fs, n)
	for i in range(n):
		ti = n-i-1      # Circular indexing of last and second-to-last bins
		vi = (n-i-2)%n
		test_fl = rr[ti]
		valid_fl = rr[vi]
		train_fls = [x for i,x in enumerate(rr) if i not in (ti, vi)]
		train_fls = [x for xs in train_fls for x in xs]
		#print(ti, vi, len(train_fls))
	
		with open(f"{dataset_dir}/{output_dir}/{part}_{face}_prot{i+1}_of_{n}.json", "w") as fd:
			json.dump({
				"prefix": f"{dataset_dir}/{part}/",
				"train": train_fls,
				"valid": valid_fl,
				"test": test_fl
			}, fd, indent=2, ensure_ascii=False)
	print(f"Saved {n}-fold protocols at {dataset_dir}/{output_dir}")

