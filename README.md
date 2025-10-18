# Pneumonia DetectionCOVID-19 CT Scan Segmentation with U-net and SAM-Adapter

This repository contains code and datasets for semantic segmentation of COVID-19 CT scans, mainly **U-net** and testing if U-net + **SAM-Adapter** models improves lesion localization or not.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [DataLoader Note](#dataloader-note)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project helps detecting sub-Pneumonia data: COVID19.This was a brutal disease that took lives of so many people. Hard working people from around the world had dedicated their works to science: COVID19 lesion annotated masks.

This project aims to segment infection regions in COVID-19 CT scans. It leverages:

- **U-net**: baseline segmentation model
- **SAM-Adapter**: improves mask quality using U-net-generated prompts
- **Custom datasets** with lung ROI preprocessing and mask normalization.

Key features:

- Automatic cropping around lung regions using lung masks
- Data augmentation and normalization pipelines
- Support for multi-class or binary masks
- Overlay visualization for qualitative analysis

---

## Dataset

The dataset is divided into:

- **Training**: 2 subsets for model training
- **Evaluation**: 1 subset for inference and testing

Preprocessing steps:

- Removal of duplicate slices
- Exclusion of slices with masks containing only background (~80%)
- Cropping to lung region (ROI)
- Normalization and mapping using **bone colormap**

> ⚠️ Some slices are low-quality or misleading; manual inspection may be required for best results.

---
