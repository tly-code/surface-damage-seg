# Surface Damage Segmentation

Deep-learning-based segmentation for Antarctic ice shelf surface damage detection using Landsat RGB imagery.

## Requirements

- Python 3.12
- CUDA 12.4

## Installation

Clone the repository and create the conda environment:

```
git clone https://github.com/tly-code/surface-damage-seg.git
cd surface-damage-seg
conda env create -f environment.yml
conda activate damage_seg
```

To use the environment in Jupyter notebooks:

```
python -m ipykernel install --user --name damage_seg --display-name "damage_seg"
```

## Acknowledgements

### Data

Landsat 7/8/9 data courtesy of the U.S. Geological Survey (USGS). This project uses imagery accessed through Google Earth Engine.

### Code

This project is built upon SegFormer (https://github.com/NVlabs/SegFormer) for semantic segmentation. We thank the authors for making their work publicly available. 

If you use this code, please cite:

```bibtex
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Project Structure

```
surface-damage-seg/
├── landsat_daily_composite_export_GEE.js      # Google Earth Engine data download script
├── damage_upload.py                           # Segmentation model definition
├── damage_inference.ipynb                     # Inference script
├── ice_shelf_damage_temporal_analysis.ipynb   # Temporal analysis
├── environment.yml                            # Conda environment specification
└── README.md
```
