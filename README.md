# Sandstone Thin-Section Annotated Sample Generation Workflow

This repository provides a compact public demonstration of a sandstone thin-section annotated sample generation workflow used in the manuscript:

**Automated Generation Method of Labeled Sandstone Thin-Section Images for Intelligent Identification**

The workflow starts from a prepared digital-core slice and generates a pair of synthetic sandstone thin-section images, including one plane-polarized light image (PPL), one cross-polarized light image (XPL), and one spatially consistent semantic label mask.

This repository is intended to demonstrate the main methodological process. It does not include the complete experimental dataset, the full three-dimensional digital-core volume, the complete mineral particle library, DDPM training code, or semantic-segmentation model training code.

---

## Workflow overview

The demonstration notebook includes the following steps:

1. Load one prepared digital-core slice and a feldspar-dissolution mask.
2. Segment mineral regions into independent particle instances.
3. Reconstruct one PPL image, one XPL image, and one semantic label mask using mineral particle textures.(Pore ​​simulation during the reconstruction phase)
4. Apply feldspar-dissolution enhancement.
5. Apply clay/cementation-rim enhancement.
6. Save the final PPL image, XPL image, and single-channel label mask as a semantic-segmentation sample.

The generated PPL image, XPL image, and label mask are spatially aligned.

---

## Repository structure

```text
Sandstone-thin-section-annotated-sample-generation-workflow/
├── 3d_rockdata_slice/
│   ├── sample1_rock_base.npy
│   └── sample1_dissolution_mask.npy
├── Mineral_Particle_Library_example/
│   ├── quartzData/
│   │   ├── cross_1/
│   │   └── single/
│   ├── feldsparData/
│   │   ├── cross_1/
│   │   └── single/
│   └── rockchipsData/
│       ├── cross_1/
│       └── single/
├── generated_dataset/
│   ├── labels/
│   ├── ppl/
│   └── xpl/
├── thin_section_generation_workflow.ipynb
├── tools.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Description of files and folders

### `thin_section_generation_workflow.ipynb`

This notebook demonstrates the complete public workflow step by step. It reads a prepared digital-core slice, performs mineral particle instance segmentation, reconstructs PPL/XPL images, applies geological feature enhancement, and saves the final images and label mask in a semantic-segmentation dataset format.

### `tools.py`

This file contains the main utility functions used by the notebook, including:

* particle instance segmentation;
* particle mask generation and smoothing;
* mineral particle texture matching;
* PPL/XPL image reconstruction;
* feldspar-dissolution enhancement;
* clay/cementation-rim enhancement;
* output saving for semantic segmentation.

### `3d_rockdata_slice/`

This folder contains prepared two-dimensional digital-core slice files used by the demonstration notebook.

The public notebook reads prepared `.npy` files rather than the original raw three-dimensional digital-core volume.

Expected files:

```text
sample1_rock_base.npy
sample1_dissolution_mask.npy
```

where:

* `sample1_rock_base.npy` is the prepared digital-core semantic slice;
* `sample1_dissolution_mask.npy` is the feldspar-dissolution mask used for geological feature enhancement.

### `Mineral_Particle_Library_example/`

This folder contains an example mineral particle texture library used for texture mapping.

The expected structure is:

```text
Mineral_Particle_Library_example/
├── quartzData/
│   ├── cross_1/
│   └── single/
├── feldsparData/
│   ├── cross_1/
│   └── single/
└── rockchipsData/
    ├── cross_1/
    └── single/
```

where:

* `cross_1/` contains cross-polarized light particle textures;
* `single/` contains plane-polarized light particle textures.

For each mineral particle, the corresponding XPL and PPL texture images should have the same file name.

### `generated_dataset/`

This folder contains the generated output samples.

```text
generated_dataset/
├── labels/
├── ppl/
└── xpl/
```

where:

* `ppl/` contains generated PPL thin-section images;
* `xpl/` contains generated XPL thin-section images;
* `labels/` contains single-channel semantic label masks.

The label masks are saved as grayscale images with pixel values:

| Value | Class                            |
| ----: | -------------------------------- |
|     0 | pore |
|     1 | quartz                           |
|     2 | feldspar                         |
|     3 | lithic fragments                 |

The files in `labels/` are the masks used for semantic-segmentation training. 

---

## Requirements

The code was developed using Python and common scientific image-processing libraries.

Recommended packages include:

```text
numpy
opencv-python
scipy
scikit-image
matplotlib
jupyter
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

Alternatively, install the main packages manually:

```bash
pip install numpy opencv-python scipy scikit-image matplotlib jupyter
```

---

## How to run

1. Clone or download this repository.

```bash
git clone https://github.com/DG24ZX/Sandstone-thin-section-annotated-sample-generation-workflow.git
cd Sandstone-thin-section-annotated-sample-generation-workflow
```

2. Install the required Python packages.

```bash
pip install -r requirements.txt
```

3. Open the notebook.

```bash
jupyter notebook thin_section_generation_workflow.ipynb
```

or use JupyterLab:

```bash
jupyter lab thin_section_generation_workflow.ipynb
```

4. Run the notebook cells in order.

5. The generated PPL image, XPL image, and label mask will be saved in:

```text
generated_dataset/
├── ppl/
├── xpl/
└── labels/
```

---

## Output format

Each generated sample contains:

```text
generated_dataset/
├── ppl/sample1.png
├── xpl/sample1.png
└── labels/sample1.png
```

The PPL image, XPL image, and label mask have the same spatial size and are pixel-aligned.

The label image is a single-channel `uint8` PNG file. Its pixel values are semantic class IDs rather than display colors.

---

## Notes

* This repository provides a simplified public demonstration of the proposed workflow.
* Only one PPL image and one XPL image are generated for each sample in the current public notebook.
* The full digital-core volume used in the manuscript is not included.
* The complete mineral particle library used in the manuscript is not included.
* DDPM training code is not included.
* Semantic-segmentation model training and testing code are not included.
* The example mineral particle library may contain manually extracted particle textures, diffusion-generated particle textures, or both.
* If texture images cannot be read on Windows, avoid placing the repository in a path containing non-ASCII characters and check that the particle texture files are common image formats such as `.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff`.

---

## Code availability

The source code for the synthetic sandstone thin-section annotated sample generation workflow is publicly available at:

```text
https://github.com/DG24ZX/Sandstone-thin-section-annotated-sample-generation-workflow
```

The complete raw digital-core data, complete mineral particle library, and full experimental dataset are not included in this repository because of file size and data-sharing restrictions.

---

## License

This code is released under the MIT License.
