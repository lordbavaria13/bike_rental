# 📚 Repository Navigation Index

Welcome! This file helps you quickly find what you need in this project.

## 🚀 Quick Links

**First time here?**
→ Read [README.md](README.md) then [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

**Want to run the code?**
→ Execute `python run_pipeline.py`

**Need detailed structure explanation?**
→ See [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

---

## 📂 Where Things Are

| What I need | Where to find it |
|---|---|
| **Quick start guide** | [README.md](README.md) |
| **Detailed structure explanation** | [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) |
| **Full analysis & results** | [docs/PROJECT_ANALYSIS.md](docs/PROJECT_ANALYSIS.md) |
| **Data encoding details** | [docs/ENCODED_DATASETS.md](docs/ENCODED_DATASETS.md) |
| **Raw data** | `data/raw/` |
| **Processed data** | `data/processed/` |
| **Data processing code** | `src/data_processing/` |
| **Model implementations** | `modelling/00_dummy_regressor/` through `modelling/08_neural_network/` |
| **Analysis & plots** | `analysis/` |
| **Research paper** | `paper/` |
| **Configuration** | `modelling/common/config.py` |

---

## 🔄 Workflow

1. **Data Processing** (`src/data_processing/`)
   - Transform raw data → processed datasets
   - Run individually or via `run_pipeline.py`

2. **Model Training** (`modelling/XX_*/`)
   - Train models on processed data
   - 9 different algorithms included
   - Results saved to `results/` subdirectories

3. **Model Comparison** (`modelling/99_model_comparison/`)
   - Compare all models
   - Generate performance rankings

4. **Analysis & Visualization** (`analysis/`)
   - Generate insights from results
   - Create presentation-ready plots
   - Output to `analysis/{plots,insights,results}/`

---

## 🎯 Common Tasks

### Run Everything
```bash
python run_pipeline.py
```

### Train a Single Model
```bash
python -m modelling.06_random_forest.train_random_forest --experiment all
```

### Process Data Only
```bash
python src/data_processing/05_create_encoded_dataset.py
```

### Generate Analysis
```bash
cd analysis/
python generate_time_error_insights.py
python plot_top3_rmse_by_split.py
```

### View Results
- Check `modelling/06_random_forest/results/` (best model)
- See `docs/PROJECT_ANALYSIS.md` (performance summary)
- Check `analysis/plots/` (visualizations)

---

## 🏆 Best Performing Models

1. **Random Forest** - RMSE: 30.31, R²: 0.406
2. **Gradient Boosting** - RMSE: 30.57, R²: 0.396
3. **Lasso Regression** - RMSE: 33.49, R²: 0.275

See [docs/PROJECT_ANALYSIS.md](docs/PROJECT_ANALYSIS.md) for complete comparison.

---

## 📖 Documentation Map

```
docs/
├── README.md                      ← Start here
├── REPOSITORY_STRUCTURE.md        ← Detailed guide  
├── PROJECT_ANALYSIS.md            ← Results & metrics
├── ENCODED_DATASETS.md            ← Data pipeline
└── (this file: INDEX.md)
```

---

## 💡 Tips

- **Lost?** Check [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)
- **Need results?** Open [docs/PROJECT_ANALYSIS.md](docs/PROJECT_ANALYSIS.md)
- **Modify code?** Start in `modelling/XX_*/train_*.py`
- **Add analysis?** Create script in `analysis/` following naming convention
- **Configuration?** Edit `modelling/common/config.py`

---

## 🔧 Requirements

- Python 3.10+
- Dependencies in `requirements.txt`
- Install: `pip install -r requirements.txt`

---

Generated: May 26, 2026
