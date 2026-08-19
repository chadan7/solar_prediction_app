# Solar Energy Yield & Performance Analytics App

An end-to-end data pipeline, machine learning model, and interactive Streamlit web application designed to process solar irradiance parameters ($GHI$, $DNI$, $DHI$), derive theoretical solar power output ($P_{theoretical}$), and predict system energy performance ratios.

### How to use the app (Installation & Execution)

Ensure Python 3.8+ is installed along with the required libraries (Open cmd and paste):

```bash
pip install streamlit pandas numpy scikit-learn geopy

```

1. **Clone the repository**:
```bash
git clone https://github.com/chadan7/solar_prediction_app.git
cd solar_prediction_app

```

2. **Run the Streamlit application**:
```bash
python -m streamlit run app.py

```
A browser with local host will be launched. The device should be connected to the internet so the app can run smoothly and fetch weather conditions from open-meteo.

### Repository Structure

```text
.
├── app.py                      # Main Streamlit application runner
├── Data/                       # Raw & processed data assets
│   ├── initial_data/           # Raw datasets sourced online
│   └── data_cleaned/           # Notebooks & generated data sheets
│       ├── Data1.ipynb
│       ├── Theoretical output calculations for station07.ipynb
│       ├── Data 2 - k an ratios added.ipynb
│       ├── station07_theoretical.csv
│       ├── station07_with_K.csv
│       └── station07_with_K_ratios.csv         # Final production dataset used for modeling
├── Model/                                # Model training resources & references
│   ├── Training.ipynb                    # Model training and validation notebook
│   └── random_forest_k_pv.pkl            # Model itself
└── codes/                    # Cleaned processing modules
    ├── code1_theoretical.py
    ├── code2_weather.py
    └── code3_pipeline.py              # Model loader & predictor pipeline

```


### Data Pipeline Breakdown

### `Data/data cleaning/`

1. **`Theoretical_output_calculations_for_station07.ipynb`**:
* Calculates theoretical solar power ($P_{theoretical}$) alongside $GHI$, $DNI$, and $DHI$ parameters using solar modeling standards.
* **Output**: `station07_theoretical.csv`


2. **`Data_1.ipynb`**:
* Merges `station07_theoretical.csv` variables into the primary workflow.
* Resolves date-time indexing issues, ensures continuous temporal sequence with no missing rows, and computes initial clearness index ($K_t$) approximations.


3. **`Data_2 - k_an_ratios_added.ipynb`**:
* Performs unit conversions and computes $K_t$ exclusively during daylight hours ($GHI > 0$).
* Filters out zero-sunlight hours, generates target feature ratios, and exports final processing artifacts.
* **Outputs**: `station07_with_K.csv` and `station07_with_K_ratios.csv` (*`station07_with_K_ratios.csv` serves as the final model input*).


### `Model/`

* **`Training.ipynb`**: Handles feature engineering, cross-validation, hyperparameter tuning, and performance evaluation.
* **`Model_link`**: Contains the external cloud storage link to download the serialized model binary (excluded from repository due to GitHub file size limits).

### `codes/`

* Clean modular scripts handling data manipulation and inference.
* **`code3_pipeline.py`**: Loads the trained model binary. **Configuration required**: Before executing, open `code3_pipeline.py` and update the local file path variable pointing to your downloaded model file.


