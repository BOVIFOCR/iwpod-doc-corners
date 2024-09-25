from glob import glob
import random
import json

import argparse
import os

if __name__ == "__main__":
	ps = argparse.ArgumentParser()
	ps.add_argument("--nfold", "-n", type=int, default=10)
	ps.add_argument("--protocol", "-p", type=str, default="closeup")
	args = ps.parse_args()

	part = args.protocol
	n = args.nfold

	if not os.path.exists("rg_dataset/prots/"):
		os.mkdir("rg_dataset/prots/")

	# This line is just for getting all labels - doesn't matter if from wild or closeup
	fs = [x.split("/")[-1] for x in glob("rg_dataset/wild/*") if not x.endswith(".txt")]
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
	
		with open(f"rg_dataset/prots/{part}_prot{i+1}_{n}.json", "w") as fd:
			json.dump({
				"prefix": f"rg_dataset/{part}/",
				"train": train_fls,
				"valid": valid_fl,
				"test": test_fl
			}, fd, indent=2, ensure_ascii=False)

