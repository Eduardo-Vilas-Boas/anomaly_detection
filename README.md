## Data

### Training datasets

Here is the list of datasets used for this project:
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects.
- [DAMG](https://zenodo.org/records/12750201) was inspired by the fact that automated optical inspection allows to reduce the cost of industrial quality control significantly. The competitors had to design a classification algorithm which:
    - detects miscellaneous defects on various statistically textured backgrounds.
    - learns to discern defects automatically from a weakly labelled training data.
    - works on data whose exact characteristics are unknown at development time.
    - adapts all parameters automatically and does not require any human intervention.
    - has a moderate running time (in this competition 24 hours for training and 12 hours for the test phase).
    - takes into account asymmetric costs for false positive and false negative decisions (1:20 was used for the competition).
- The [Lusitano](https://kailashhambarde.github.io/Lusitano/) dataset comprises a comprehensive selection of fabrics and defects sourced from a reputable textile company based in Portugal.
- The [Aitex fabric image database](https://data.mendeley.com/datasets/663j22s43c/3) is consisted of 245 images of 7 different fabrics with image sizes of 4096×256 pixels. There were 140 defect-free images, 20 for each type of fabric. In each of the defected category, there were 105 images.

### Test datasets

- [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) - Six kinds of typical surface defects of the hot-rolled steel strip are collected, i.e., rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In) and scratches (Sc). The database includes 1,800 grayscale images: 300(split into 240 images for training and 60 images for testing.) samples each of six different kinds of typical surface defects.

## Data processing

conda env create --file=conda_env.yaml
