# 🌫️ Air Pollution Modeling using Gaussian Process Regression

### ETH Zurich – Probabilistic Artificial Intelligence (PAI) 2025

---

## 📖 Overview

This project models **fine particulate matter (PM2.5)** concentrations using **Gaussian Process Regression (GPR)** to support urban planning decisions.  
Given air quality measurements from mobile stations, the goal is to **predict PM2.5 levels at unmeasured locations** and identify low-pollution areas suitable for residential development.

---

## 🎯 Key Ideas

- **Gaussian Process Regression:**  
  Models spatial air pollution using a Matern kernel with noise and scaling terms.

- **Scalable Learning:**  
  Uses *MiniBatchKMeans* clustering to subsample training data efficiently for large-scale GPR.

- **Asymmetric Cost Function:**  
  Penalizes underestimation more heavily in residential zones to ensure conservative predictions.
