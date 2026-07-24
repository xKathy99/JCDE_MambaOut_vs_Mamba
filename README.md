
<a id="readme-top"></a>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#project-structure">Project Structure</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#segmentation-pipeline-detection">Run MedSAM Detection for Segmentation</a></li>
        <li><a href="#segmentation-pipeline-medsam-segmentation">Run Segmentation Pipeline (MedSAM)</a></li>
        <li><a href="#run-mamba-mambaout-variants-for-segmentation">Run Mamba and MambaOut U-Net Variants for Segmentation</a></li>
        <li><a href="#visualisation-prerequisites">Run Visualisation</a></li>
      </ul>
    </li>
  </ol>
</details>



## About The Project

GitHub repository for A Comprehensive Benchmark of Mamba vs. MambaOut with U-Net for Cardiac MRI Segmentation: A Cross-Dataset Evaluation and Analysis 


## Project Structure
jcde_unet/                                  # Submission to JCDE Journal
   ├── data/    
   ├── datasets/                            # link-to-datasets.txt                       
   ├── checkpoints/                         # llink-to-weights.txt          
   ├── models/
   │   ├── layers.py
   │   ├── hybrid_mamba_unet.py             # hybrid Mamba U-Net         
   │   ├── hybrid_mambaout_unet.py          # hybrid MambaOut U-Net
   │   ├── pure_mamba_unet.py               # pure Mamba U-Net
   │   ├── pure_mambaout_unet.py            # pure MambaOut U-Net
   │   ├── pretrained_mamba_enc_unet.py     # pre-trained Mamba Encoder U-Net
   │   └── pretrained_mambaout_enc_unet.py  # pre-trained MambaOut Encoder U-Net                         
   ├── utils/    
   ├── losses/                             
   ├── metrics/     
   ├── requirements.txt            
   ├── config.py
   ├── test.py
   ├── train.py
   └── pathology_analysis.py


### Run Mamba Mambaout Variants for Segmentation

### Installation

```sh
   git clone https://github.com/mmlchang/2023_FRGS_HeartDigitalTwin.git
   cd jcde_unet
   pip install -r requirements.txt
```
### Run Testing
#### 1. Download model weights in /checkpoints and cardiac datasets to /dataset (link provided in directory)

#### 2. Open config.py and set the weights and MODEL_ARCH variables to one of the six available variants

   ```sh
      # config.py
      WEIGHTS_DIR       = "../checkpoints/model"
      BEST_WEIGHTS_DIR  = "../checkpoints/model/best"

      MODEL_ARCH = "pretrainedmambaout"  
      # options:
      # pretrainedmamba, hybridmamba, hybridmambaout
      # puremamba, puremambaout
   ```

   #### 3. Run test.py

   ```sh
      python test.py --dataset acdc     # ACDC test set
      # OR python test.py --dataset mms      # M&Ms test set
      # OR python test.py --dataset mms2     # M&Ms-2 test set
   ```

   #### 4. Run Pathology Analysis
   Change the info_csv variable in pathology_analysis.py before running
   ```sh
      python pathology_analysis.py --dataset acdc # OR mms OR mms2
   ```

   ### Run Training
   ```sh
      python train.py    # default trained on ACDC test set
   ```

   ---


<p align="right">(<a href="#readme-top">back to top</a>)</p>

