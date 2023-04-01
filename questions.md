### General issues (need more thinkings)

1. In general, I don't think over method is elegant. Things that I will try first:
   - Remove pretraining on C-GAN and see experimental results
   - Remove distance constraint and see experimental results

### Experiment issues

1. Why balanced regime over random regime? still not convinced
2. For imbalanced regime:
   - How to pick the class that is seen in general?
   - Probably pick 3 or 5 classes for MNIST/FashionMNIST experiments; I think 3 is good enough for experimental purpose.

### Training issues

1. Model Architectures
2. **Tuning** for both pretraining and actual training stages
3.

### TODO

1. Start writing first; setup everything
2. Think about contributions

   - Exploit the advantages of limited OoD samples

3. Design experiments based on purposes

## Contribution

1. Classic approach: no samples
   - ODIN, MaHa
2. New: exploit advantanges of few samples
   - AUX (if necessary)
   - WOOD
3. Better within-class generalization
4. Better separability - (significant improvement at high TNR)
5. Data augmentation


