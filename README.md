# ENGE707 Assessment 3: Cat and Dog Facial Feature Classification

Deep learning pipeline that classifies cropped cat and dog facial features into eight classes
(CatEye, DogEye, CatFace, DogFace, CatMouth, DogMouth, CatNose, DogNose).

| Result | Accuracy |
| --- | --- |
| Validation, fine-tuned ResNet-50 | 95.48% |
| Validation, final method (ensemble + TTA + shape rule) | 99.76% |
| Kaggle public leaderboard, first submission | 92.42% |
| Kaggle public leaderboard, final submission | **95.84%** |

Kaggle username: `christiancantos`

## Repository structure

```
├── README.md
├── requirements.txt
├── data/                      # Kaggle files go here (not included, see Setup)
├── src/
│   ├── KaggleCompetition.ipynb   # main notebook, one section per assessment task (1 to 6)
│   ├── preprocessing.py       # paths, class mapping, transforms, Dataset and DataLoaders
│   ├── models.py              # baseline CNN and ResNet-50 definitions
│   └── training.py            # training loop, evaluation, prediction, submission writer
├── outputs/
│   ├── splits/                # train/validation/test lists, class mapping, excluded images
│   ├── models/                # trained weights (.pt)
│   ├── history/               # per-epoch training logs (learning curves)
│   ├── probs/                 # cached model probabilities used in Task 6
│   ├── figures/               # all figures used in the report
│   ├── quality_metrics.csv    # per-image measurements from Task 1
│   ├── submission.csv         # first Kaggle submission (fine-tuned ResNet-50)
│   └── submission_improved.csv   # final Kaggle submission
└── report/                    # final report and leaderboard screenshot
```

## Setup

1. **Python 3.12 or later** with a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Data:** download the competition files from Kaggle and extract them into `data/`, so it contains
   `train/`, `test/`, `train_labels.csv` and `sample_submission.csv`. Nested folders from the Kaggle zip
   (for example `data/train/train/`) are fine; images are found recursively.
3. **macOS only:** if downloading the pretrained ResNet-50 weights fails with `CERTIFICATE_VERIFY_FAILED`,
   run `Install Certificates.command` from the Python folder in Applications, then restart the kernel.

All paths are found automatically by `preprocessing.py`, which walks up from `src/` to the folder
containing `data/`. No paths need editing.

## Running the notebook

Open `src/KaggleCompetition.ipynb` and run the sections in order. A fixed random seed (42) is used throughout.

| Section | What it does | Approximate time |
| --- | --- | --- |
| Task 1 | Data structure checks, class distribution, data quality | 5 to 10 min |
| Task 2 | Cleaning, stratified 80/20 split, leakage and pipeline checks | 2 min |
| Task 3 | Model definitions, input/output checks, single-batch sanity test | 2 min |
| Task 4 | Three training runs, learning curves, diagnosis | seconds if weights exist; about 2.5 h to retrain |
| Task 5 | Validation metrics, confusion matrix, first submission | 10 to 15 min |
| Task 6 | Error analysis, improvement experiments, final submission | seconds if `outputs/probs/` exists |

**Reproducing without retraining (recommended).** With the saved weights in `outputs/models/`, Task 4 loads
each run instead of training it, so the whole notebook reproduces the reported results and both submission
files without a GPU.

**Retraining from scratch.** Set `retrain=True` in the three run cells of Task 4 (run them in order, since
run 3 starts from run 2's weights) and delete `outputs/probs/` before Task 6. Training used an Apple M3 GPU
(`mps`); CUDA and CPU are also supported. GPU training is not bit-for-bit deterministic, so retrained
accuracies may differ slightly (typically by a few tenths of a percentage point) from those reported.

## Model weights

[If the ResNet-50 weight files are not in the repository, state here where they are, for example:
"The ResNet-50 weights (about 94 MB each) are attached to this repository's Releases page. Download
`run1_baseline_cnn.pt` and `run3_resnet50_finetune.pt` into `outputs/models/`."]

## References

Libraries and papers are listed in the report's reference list. Main libraries: PyTorch, torchvision,
scikit-learn, pandas, NumPy, Pillow, Matplotlib and SciPy. ResNet-50 ImageNet weights from torchvision
(`ResNet50_Weights.IMAGENET1K_V2`).