# LCC4CD and Dynamic Norway

This repository currently contains the code for the pipeline of the LCC4CD system and the associated Dynamic Norway model, tied to a master's thesis in Computer Science.

The master's thesis was written in collaboration with [NINA (Norwegian Institute for Nature Research)](https://www.nina.no).

There is no script available to run the whole pipeline. However, subscripts are available to run the separate parts of the pipeline.

---

### 📥 Data Download

All code is based on data downloaded through GEE (Google Earth Engine) using the scripts in the [`data_generation/`](./data_generation/) folder.  
The project ID has to be linked to your personal GEE account.

- Training data can be downloaded using [`export_baseline_training_data.py`](./data_generation/export_baseline_training_data.py)
- AOI inference data can be downloaded using [`export_data_aoi.py`](./data_generation/export_data_aoi.py)
- NIBIO Summer dataset can be downloaded using [`export_NIBIO_summer.py`](./data_generation/export_NIBIO_summer.py)

---

### 🔄 Pipeline

The pipeline is run in two main steps, once all required data has been downloaded.  
Make sure all constants are configured correctly before running any part of the pipeline.

1. A model is trained using [`training.py`](./training.py)
2. Change detection and evaluation is done through [`built_probability.py`](./built_probability.py)


### 👩‍💻 About the Authors

This repository and all included scripts were created as part of our Master's thesis work at NTNU,  
in collaboration with [NINA](https://www.nina.no).

**Ingvild Almåsbakk** & **Eva Anette Johansen**  
July 2025
