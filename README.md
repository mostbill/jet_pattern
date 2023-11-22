# jet_pattern

## Introduction

Here stores the code that we use VLASS(Very Large Array SKy Survey) public rado images to automatically identify blazar-like or non-blazar-like sources from a pure morphological approach. For details, refer to the relevant publication (under review).

## Usage

Prepare a folder with all fits file in it. The algorithm will:

1. Classify each epoch into a morphological class
2. Download reference photometry with the order of: DECaLS DR9 (rz) - PanSTARRS (rz) - SkyMapper (rz), neoWISE (W1)
3. Generate a diagnostic PDF for each source

An example line of code to kick it off:

```python
python pattern.py --mode multiple --pdf True --folder low_mass_agn --catalog /Path/to/Catalog
```
## Contact
Zhang-Liang Xie
email: xie[at]mpia.de

## File tree
main code: _pattern.py_

output tables:_output_final_
