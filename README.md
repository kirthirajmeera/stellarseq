# 🛸 StellarSeq - Astronaut Genetic Stress Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stellarseq.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**AI-powered tool for predicting astronaut stress severity from gene expression data using NASA spaceflight genomics.**

🔗 **Live App:** [https://stellarseq.streamlit.app](https://stellarseq.streamlit.app)

---

## 🚀 Overview

StellarSeq is a machine learning application that predicts genetic stress responses in astronauts by analyzing gene expression data from NASA's Open Science Data Repository (OSDR). Built for researchers, students, and space biology enthusiasts.

### Key Features

- **🧬 Raw Data Processing** — Upload NASA OSDR expression matrices directly
- **🎯 Single Sample Prediction** — Manual input for individual analysis
- **📡 Batch Processing** — Analyze multiple samples simultaneously
- **📊 Interactive Visualizations** — Stress gauges, histograms, and feature importance charts
- **📥 Export Results** — Download predictions as CSV

---

## 📸 Screenshots

### Dashboard
![StellarSeq Dashboard](https://via.placeholder.com/800x400?text=Dashboard+Screenshot)

### Prediction Results
![Prediction Results](https://via.placeholder.com/800x400?text=Results+Screenshot)

### Batch Analysis
![Batch Analysis](https://via.placeholder.com/800x400?text=Batch+Screenshot)

> 💡 *Replace placeholder images with actual screenshots of your app*

---

## 🔬 Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | XGBoost Regressor |
| **R² Score** | 0.974 |
| **RMSE** | 2.92 |
| **MAE** | 2.09 |
| **Training Samples** | 112 |
| **Features** | 14 |

---

## 📊 Data Sources

Trained on gene expression data from **9 NASA spaceflight studies**:

- **Organisms:** Mouse (*Mus musculus*), Human (*Homo sapiens*)
- **Tissues:** Thymus, Liver, Muscle, Blood, Retina, Skin
- **Conditions:** Spaceflight, Radiation (HZE), Hindlimb Unloading, Ground Control

Data sourced from [NASA OSDR](https://osdr.nasa.gov/bio/repo/search) (Open Science Data Repository).

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Framework:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Deployment:** Streamlit Cloud

---

## 📁 Project Structure

```
stellarseq/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── model/
│   ├── best_model.pkl     # Trained XGBoost model
│   ├── scaler.pkl         # Feature scaler
│   └── training_metadata.json
├── .streamlit/
│   └── config.toml        # Streamlit theme configuration
└── README.md
```

---

## 🚀 Local Development

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/stellarseq.git
cd stellarseq

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Model Files Required

Copy these files to the `model/` directory:

```bash
# From your trained model directory
cp path/to/best_model.pkl model/
cp path/to/scaler.pkl model/
cp path/to/training_metadata.json model/  # optional
```

---

## 📖 How to Use

1. **Visit** [stellarseq.streamlit.app](https://stellarseq.streamlit.app)
2. **Choose input method:**
   - **Single Prediction:** Enter sample characteristics manually
   - **Batch Upload:** Upload NASA OSDR expression matrix (CSV/TSV)
3. **Click "Analyze"** to get stress predictions
4. **Download results** as CSV

### Getting NASA Data

See the **Help** tab in the app for detailed instructions on downloading data from [NASA OSDR](https://osdr.nasa.gov/bio/repo/search).

---

## 🧪 Sample Studies for Testing

| Study ID | Organism | Condition | Link |
|----------|----------|-----------|------|
| OSD-4 | Mouse | Spaceflight | [View](https://osdr.nasa.gov/bio/repo/data/studies/OSD-4) |
| OSD-87 | Mouse | Spaceflight | [View](https://osdr.nasa.gov/bio/repo/data/studies/OSD-87) |
| OSD-137 | Mouse | Spaceflight | [View](https://osdr.nasa.gov/bio/repo/data/studies/OSD-137) |
| OSD-13 | Human | Spaceflight | [View](https://osdr.nasa.gov/bio/repo/data/studies/OSD-13) |

---

## 📚 Keywords

`NASA` `spaceflight` `gene expression` `transcriptomics` `astronaut health` `stress prediction` `machine learning` `XGBoost` `bioinformatics` `GeneLab` `OSDR` `space biology` `genomics` `RNA-seq` `differential expression`

---

## 👩‍💻 Author

**Meera Kirthiraj**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/meerakirthiraj/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [NASA Open Science Data Repository](https://osdr.nasa.gov/) for spaceflight genomics data
- [NASA GeneLab](https://genelab.nasa.gov/) Analysis Working Group
- [Streamlit](https://streamlit.io/) for the web framework

---

<p align="center">
  <strong>⭐ Star this repo if you find it useful!</strong>
</p>
