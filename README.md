## Deep Learning for Generating Phase-Conditioned Infrared Spectra

Infrared (IR) spectroscopy is an efficient method for identifying unknown chemical compounds. To accelerate the chemical analysis of IR spectra, various calculation and machine learning methods for simulating IR spectra of molecules have been studied in chemical science. However, existing calculation and machine learning methods assumed a rigid constraint that all molecules are in the gas phase, i.e., they overlooked the phase dependency of the IR spectra. In this paper, we propose an efficient phase-aware machine learning method to generate phase-conditioned IR spectra from 2D molecular structures. To this end, we devised a phase-aware graph neural network and combined it with a transformer decoder. To the best of our knowledge, the proposed method is the first IR spectrum generator to generate the phase-conditioned IR spectra of real-world complex molecules. The proposed method outperformed state-of-the-art methods in the tasks of generating IR spectra on a benchmark dataset containing experimentally measured 11,546 IR spectra of 10,288 unique molecules.

## Run
- exec.py: A python script to optimize model parameters of PASGeN on the NIST dataset.

## Datasets
<code style="color : red">The collected NIST dataset will be available for appropriate requests.</code>
