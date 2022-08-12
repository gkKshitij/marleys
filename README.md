# Python script/algos for ImageRecognition

## For running the script

**REQUIRES CONDA INSTALLED**

1. `conda env create -n pyis -f env.yml`
2. `conda activate pyis`
3. `python test.py`

---

---

## Current task: To detect and calculate measurements of tiles from images and compare with mentioned ones _[with respect to tiledim.png]_

### Expected models and other technologies

- **Programming Language**: Python
- **Model**: CNN subtypes would be overkill as harr-cascades with a little augmentation must be enough

- **Variables that might differ** (break the working)
  - position of camera/ object
  - distance between camera and object (tile)
  - absence of reference object (like paper)
  - shadows/lighting

### Expected Steps

> For measurement of dimensions of tiles

- [ ] Get image/video from webpage [UI related]
- [x] Crop Region Of Interest [ROI]
- [x] Carry out various process to get proper contours
  - [x] Gray
  - [x] Blur (7x7 Gaussian Blur)
  - [x] Edge (Canny mid range (30,150))
  - [x] Thresh
  - [x] Dilate
  - [x] Erode
- [x] Measure the dimensions of various parts of contours according to tiledim.png.

##### **TODO: (could me majorly improved)**

<!-- ![tiledim.png](IMG_1097%20-%20pre-re-sized.JPG) -->

#### This might be a little tricky considering our contour will have many points so the current implementation is measuring the other contours of the tile and not the actual tile but it actually giving similar effect.

---

---

---

### Other (not that important right now neither we have the dataset so must be extracted from the broucher)

types of roof tiles: https://www.newenglandmetalroof.com/types-of-roof-tiles/
and others are mentioned in the company brochure.

#### Use case #2 is to Detect Flaws and discrepancies in the structure of roof tiles manufactured, with help of image recognition and also recognize the type of tiles

### For detection and type segregation

- [x] Get images of roofs (tiles) design etc.
- [ ] Augment images of defected roofs (tiles) design etc.
- [ ] Label the dataset if it isn't already.
- [ ] Train the model to detect discrepancies/deformities.
- [ ] Highlight areas with problems and confidence percentage.
