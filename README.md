# DMG-Net
DMG-Net: A Dual-Encoder Network with Mamba and Channel Graph Fusion for Retinal Vessel Segmentation

Accurate segmentation of retinal vessels is critical for ocular disease diagnosis but remains challenging due to the intricate morphology of thin vessels, low contrast against the background, and the inherent trade-off between local detail preservation and global connectivity. To address these limitations, we propose DMG-Net, a novel dual-encoder network integrating Mamba and graph-based fusion. Specifically, we design a dual-path encoder. An Edge-Guided Extraction (EGE) block utilizes Ghost convolutions to efficiently extract high-frequency vascular details while suppressing redundancy, and a Multi-Directional Selective State-Space (MDSS) module leverages the Mamba architecture to capture long-range dependencies with linear computational complexity. To bridge the semantic gap in skip connections, we introduce a Channel Graph Fusion (CGF) module that learns inter-channel topology for more discriminative feature fusion. Furthermore, a Multi-Scale Integration (MSI) block adaptively fuses hierarchical decoder features to mitigate information loss during upsampling. Extensive experiments on CHASE_DB1, DRIVE, and STARE demonstrate that DMG-Net shows consistent performance improvements over several recent methods, particularly in detecting thin and low-contrast vessels. The results demonstrate the effectiveness of the proposed dual-path global-local encoding and graph-guided multi-scale fusion strategy for retinal vessel segmentation.

Datasets
DRIVE (Digital Retinal Images for Vessel Extraction) dataset is available at: https:/drive.grand-challenge.org/
STARE (Structured Analysis of the Retina) dataset is available at: http://cecas.clemson.edu/\string~ahoover/stare/
CHASE_DB1 (Child Heart and Health Study in England) retinal vessel reference dataset is available at: https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/

<img width="1872" height="2253" alt="FULL" src="https://github.com/user-attachments/assets/57f67fbc-1441-4771-b74e-84233671ad40" />

<img width="2009" height="1091" alt="comparison" src="https://github.com/user-attachments/assets/92087268-81a1-4b1f-8764-7f8692ea33f2" />
