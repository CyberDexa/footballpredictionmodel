# Knowledge Check: Model Training

Test your understanding of the model training pipeline.

---

### Question 1: Data Source

Why did we choose OpenFootball over API-Football?

a) OpenFootball is faster  
b) OpenFootball has the current 2025-26 season data for free  
c) OpenFootball has more detailed statistics  
d) OpenFootball requires an API key

<details>
<summary>See Answer</summary>

**b) OpenFootball has the current 2025-26 season data for free**

API-Football's free tier only provides data from 2021-2023, which is useless for predicting current matches. OpenFootball is community-maintained and includes up-to-date match results.
</details>

---

### Question 2: Feature Engineering

What is "data leakage" and how do we prevent it?

a) When data is lost during transfer - we use checksums  
b) When future information is used to predict the past - we filter by `date < current_date`  
c) When data is duplicated - we use deduplication  
d) When API limits are exceeded - we use caching

<details>
<summary>See Answer</summary>

**b) When future information is used to predict the past - we filter by `date < current_date`**

Every feature calculation function uses `df[df['Date'] < date]` to ensure we only use information that would have been available BEFORE the match. This prevents the model from "cheating" by seeing future results.
</details>

---

### Question 3: Why 37 Features?

What are the 4 main categories of features we create?

<details>
<summary>See Answer</summary>

1. **Team Form Features** (14) - Last 5 matches performance for both teams
2. **Home/Away Specific Form** (6) - Venue-specific performance
3. **Head-to-Head Stats** (4) - Historical matchups between the teams
4. **Season Statistics** (8) - Current season standings and performance
5. **Derived Features** (5) - Differences between teams (form_diff, goals_diff, etc.)

Total: 37 features
</details>

---

### Question 4: Model Selection

We train 3 different model types. Which one typically wins and why?

a) Random Forest always wins because it has more trees  
b) Logistic Regression always wins because it's fastest  
c) Different models win for different targets - there's no single winner  
d) Gradient Boosting always wins because it learns sequentially

<details>
<summary>See Answer</summary>

**c) Different models win for different targets - there's no single winner**

Random Forest often wins for over/under predictions (binary classification), while Gradient Boosting tends to perform better for match result (3-class). That's why we train all three and pick the best per target.
</details>

---

### Question 5: Time-Series Split

Why do we use TimeSeriesSplit instead of random train/test split?

a) It's faster to compute  
b) Football is time-series data - we must train on past to predict future  
c) Random splits use more memory  
d) sklearn doesn't support random splits

<details>
<summary>See Answer</summary>

**b) Football is time-series data - we must train on past to predict future**

In real use, we predict future matches using past data. Random splits would mix future matches into training data, giving unrealistically high accuracy that wouldn't generalize to real predictions.
</details>

---

### Question 6: Model Persistence

Why do we save 7 files per league?

<details>
<summary>See Answer</summary>

We save:
1. `{league}_scaler.joblib` - The StandardScaler fitted to training data
2. `{league}_match_result_model.joblib` - 3-class prediction model
3. `{league}_home_win_model.joblib` - Binary home win model
4. `{league}_away_win_model.joblib` - Binary away win model
5. `{league}_over_1.5_model.joblib` - Binary over 1.5 goals model
6. `{league}_over_2.5_model.joblib` - Binary over 2.5 goals model
7. `{league}_metadata.joblib` - Feature column names and target configs

The scaler is crucial - new features must be scaled the same way as training features!
</details>

---

### Question 7: Pre-Training

Why do we train locally instead of on Streamlit Cloud?

<details>
<summary>See Answer</summary>

Training 19 leagues takes 10-20 minutes total. Streamlit Cloud has execution timeouts that would kill the process. By pre-training locally and committing the .joblib files to Git, the deployed app simply loads the models and makes predictions instantly.
</details>

---

## Score Yourself

- **7/7**: Expert level! You understand the full pipeline.
- **5-6/7**: Great understanding, review the modules you missed.
- **3-4/7**: Good start, re-read the modules for deeper understanding.
- **0-2/7**: No worries! Go through each module again with the code open.
