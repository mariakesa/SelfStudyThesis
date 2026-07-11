global R²:
    How much of all real matrix variance is explained by synthetic matrix?

per-neuron R²:
    For each neuron, how well does class-synthetic activity predict
    its real image tuning over 118 images?

per-image correlation:
    For each image, how similar is the real population vector to the
    synthetic population vector?

PCA comparison:
    PCA(real)
    PCA(synthetic)
    synthetic projected into real PCA basis
    score correlations per PC
    subspace angles between real and synthetic top-k PCs

contrastive PCA:
    C_real - alpha * C_synthetic
    C_synthetic - alpha * C_real
    
https://chatgpt.com/g/g-p-6a40138926d08191a609eb40d7043401-writingthesis/c/6a440738-295c-83ed-8f6d-36201de1e2b1