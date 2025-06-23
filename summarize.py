import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="details")
parser.add_argument('--name', type=str)
args = parser.parse_args()

name = args.name

if name in ['3DPC-R1', '3DPC-R2']:
    N = [100, 200, 500, 1000, 1500, 2000]
else:
    N = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


AUCs = []
TPR95 = []
TPR99 = []
for n in N:
    file_path = os.path.join('checkpoint', 'log', args.name, f'log-{n}.txt')
    with open(file_path, 'r') as f:
            lines = f.readlines()
            # print(lines[index])
            tpr95 = float(lines[-13].split(" ")[1].strip())
            tpr99 = float(lines[-9].split(" ")[1].strip())
            auc = float(lines[-5].split(" ")[1].strip())
            # print(tpr95, tpr99, auc)

    AUCs.append(auc)
    TPR95.append(tpr95)
    TPR99.append(tpr99)


print(f"Summary for {args.name} experiments")
print(f"N: {N}")
print(f"AUCs: {', '.join(f'{f:.4f}' for f in AUCs)}")
print(f"TPR95: {', '.join(f'{f:.4f}' for f in TPR95)}")
print(f"TPR99: {', '.join(f'{f:.4f}' for f in TPR99)}\n\n")

