# Water Quality Predictor

A machine learning-powered Streamlit web application for predicting and visualizing water quality using environmental data.
[requirements.txt](https://github.com/user-attachments/files/22047380/requirements.txt)

## Features

- Interactive web interface built with Streamlit
- Upload and analyze your own water quality datasets (CSV)
- Visualize data distributions and correlations
- Predict water quality using multiple machine learning models (Logistic Regression, Random Forest, Isolation Forest)
- Download results and model outputs

## Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://water-quality-predictor-esxx6mbwmkgs6esbe8hv3i.streamlit.app/)

## Getting Started

### Prerequisites

- Python 3.8+
- See `requirements.txt` for required packages

### Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/YOUR-USERNAME/water-quality-predictor.git
   cd water-quality-predictor
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the app:
   ```sh
   streamlit run code.py
   ```

## Usage

- Upload your CSV data file when prompted
- Explore visualizations and predictions in the app

## Project Structure

```
├── code.py               # Main Streamlit app
├── requirements.txt      # Python dependencies
├── Watera.csv              # Example data file (add your own)
├── README.md             # Project documentation
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.

# Requirements for Water Quality Predictor
# Install with: pip install -r requirements.txt

streamlit>=1.0.0
pandas>=1.0.0
numpy>=1.19.0
seaborn>=0.11.0
matplotlib>=3.3.0
joblib>=1.0.0
scikit-learn>=0.24.0
