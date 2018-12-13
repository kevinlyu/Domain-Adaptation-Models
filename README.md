# Domain-Adaptation-Models
my implemented domain adaptation model using pytorch


# Wasserstein Adversarial Domain Adaptation Model for Image Recognition

In this respository, we implmented our proposed Wasserstein adversarial domain adaptation (WADA) model for object recognition. Download link for dataset used to evaluate the model can be find in "data description" section.

Our WADA model consists of source extractor, target extractor, task classifier, feature relator, and domain discriminator. Since optimal transport based metric like Wasserstein distance can avoid gradient vanishing caused in $f$-divergence, we apply it in adversarial domain adaptation model to 

# Data Description 
- Digit dataset
    - MNIST
    - MNISTM
    - USPS
- Office dataset
    - Amazon
    - DSLR
    - Webcam  

# Experimental Settings

- Framework
    - Pytorch 0.4.1

- Hardware
    - CPU: Intel Core i7-7700 @3.60 GHz
    - RAM: DDR3 2400MHz 64GB
    - GPU: Geforce GTX 1080Ti 11GB
    - CUDA Version: 9.0.176
    - CuDNN Version: 7.1.1

# Experimental Results

