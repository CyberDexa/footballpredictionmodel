# Learning Path: Model Training Pipeline

> Generated: 2025-12-29
> Topic: How we achieved ML model training
> Estimated Time: 20 minutes

## Overview
- **What was built**: A multi-target ML prediction system that trains 5 different models for each of 19 football leagues
- **Why it matters**: Enables accurate predictions for match results, home/away wins, and over/under goals
- **Prerequisites**: Basic understanding of Python, machine learning concepts

## Modules
1. [01-data-pipeline.md](./01-data-pipeline.md) - 5 min - How data flows from source to training
2. [02-feature-engineering.md](./02-feature-engineering.md) - 5 min - The 37 features we create
3. [03-model-architecture.md](./03-model-architecture.md) - 5 min - The ML models and ensemble approach
4. [04-training-process.md](./04-training-process.md) - 5 min - How training actually works

## After Learning
- [knowledge-check.md](./knowledge-check.md) - Test your understanding
- [summary.md](./summary.md) - Quick reference card

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA FETCHING (OpenFootballFetcher)                         │
│     - Fetches from openfootball.github.io                       │
│     - Gets 2-4 seasons of match data per league                 │
│     - Saves to data/{league}_data.csv                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. FEATURE ENGINEERING (FeatureEngineer)                       │
│     - Creates 37 features per match                             │
│     - Team form, head-to-head, season stats                     │
│     - Derived features (form diff, goals diff)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. MODEL TRAINING (EPLPredictor)                               │
│     - Trains 3 model types per target                           │
│     - 5 prediction targets total                                │
│     - Uses TimeSeriesSplit cross-validation                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. MODEL PERSISTENCE                                           │
│     - Saves best model per target                               │
│     - 7 files per league (5 models + scaler + metadata)         │
│     - 133 total model files for 19 leagues                      │
└─────────────────────────────────────────────────────────────────┘
```
