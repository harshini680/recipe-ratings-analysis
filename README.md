# Recipe for Success: What Drives Recipe Ratings on Food.com

**Name:** Harshini Sangaraju

A data science investigation for DSC 80 at UCSD.

---

## Introduction

This project analyzes the food.com Recipes and Ratings dataset, which contains recipes and user reviews collected from food.com. It was originally compiled for recommender systems research and spans all recipes and reviews submitted since 2008. After merging recipes with their associated ratings, the dataset contains **83,782 recipes**, of which **81,173** received at least one rating.

**Central question:** What recipe characteristics — including nutritional content, complexity, and cooking time — are associated with higher average ratings?

This question is worth investigating because a recipe's average rating is the most direct signal of user satisfaction. Understanding which measurable features predict that satisfaction has practical implications for recipe developers, food content creators, and recommendation systems. It also raises a behavioral question: do users rate recipes based on objective quality indicators, or are ratings driven primarily by self-selection and subjective taste?

The columns most relevant to this investigation are:

| Column | Description |
|:---|:---|
| `name` | Recipe name |
| `minutes` | Total preparation and cooking time (minutes) |
| `n_steps` | Number of steps in the recipe |
| `n_ingredients` | Number of ingredients required |
| `tags` | Food.com tags (e.g., 'easy', 'healthy', 'dessert') |
| `calories` | Total calorie content, parsed from the `nutrition` column |
| `protein_pdv` | Protein as a percentage of daily value |
| `avg_rating` | Mean user rating across all reviews (our response variable) |

---

## Data Cleaning and Exploratory Data Analysis

**Data cleaning steps:**

1. **Left-merged** the recipes and interactions datasets on recipe ID so every recipe is retained even if it received no reviews.
2. **Replaced all ratings of 0 with `NaN`**: food.com does not allow zero-star submissions, so a rating of 0 indicates no rating was submitted — treating it as missing is appropriate.
3. **Computed `avg_rating`** as the mean of all valid (non-zero) ratings per recipe and merged it back into the recipes dataset.
4. **Parsed the `nutrition` column** from a string into individual numeric columns: `calories`, `total_fat_pdv`, `sugar_pdv`, `sodium_pdv`, `protein_pdv`, `saturated_fat_pdv`, and `carbohydrates_pdv`.
5. **Converted `submitted`** to a `pd.Timestamp` for time-based analyses.

The first five rows of the cleaned dataset are shown below:

| name | minutes | n_steps | n_ingredients | calories | avg_rating |
|:---|---:|---:|---:|---:|---:|
| 1 brownies in the world best ever | 40 | 10 | 9 | 138.4 | 4.0 |
| 1 in canada chocolate chip cookies | 45 | 12 | 11 | 595.1 | 5.0 |
| 412 broccoli casserole | 40 | 6 | 9 | 194.8 | 5.0 |
| millionaire pound cake | 120 | 7 | 7 | 878.3 | 5.0 |
| 2000 meatloaf | 90 | 17 | 13 | 267.0 | 5.0 |

**Univariate analysis:** The distribution of average ratings is strongly left-skewed — the vast majority of recipes are rated 4 or 5 stars. This reflects the self-selection inherent in the data: users rate recipes they chose to make, and people generally cook recipes they expect to enjoy.

<iframe src="assets/rating_distribution.html" width="800" height="500" frameborder="0"></iframe>

**Bivariate analysis:** The box plots below compare rating distributions across recipe complexity groups defined by number of steps. The median rating is approximately 5 across every group, with nearly identical interquartile ranges — recipe complexity has minimal relationship with how users rate a recipe.

<iframe src="assets/rating_by_steps.html" width="800" height="500" frameborder="0"></iframe>

**Interesting aggregate:** The pivot table below shows mean average rating by calorie quartile (rows) and step-count bucket (columns). Every cell sits between 4.60 and 4.68 — neither caloric content nor complexity meaningfully moves the average rating. This near-uniform table foreshadows the low signal-to-noise ratio our predictive model will encounter.

| cal_quartile | 1–5 | 6–10 | 11–15 | 16–20 | 21+ |
|:---|---:|---:|---:|---:|---:|
| Q1 (low cal) | 4.658 | 4.621 | 4.617 | 4.647 | 4.679 |
| Q2 | 4.640 | 4.613 | 4.616 | 4.625 | 4.645 |
| Q3 | 4.625 | 4.619 | 4.612 | 4.631 | 4.621 |
| Q4 (high cal) | 4.598 | 4.612 | 4.625 | 4.648 | 4.655 |

---

## Assessment of Missingness

**NMAR analysis:**

We believe the `description` column is plausibly **NMAR** (Not Missing At Random). A recipe's description is absent when the contributor chose not to write one, and that choice likely depends on what the description would have contained — for instance, contributors may omit it when they feel the recipe is self-explanatory or when they have nothing meaningful to add. Because the decision to omit depends on the unobserved content of the description rather than any recorded variable, the missingness is NMAR. To reframe it as MAR, we could collect contributor-level data such as how often that contributor writes descriptions across all their recipes, which might explain the omission through an observed variable.

**Missingness dependency:**

We analyzed the missingness of `avg_rating` — the column with the most non-trivial missingness (2,609 missing values, 3.1% of recipes). A recipe has a missing `avg_rating` when no user submitted a valid rating for it.

We performed permutation tests (1,000 simulations; test statistic = absolute difference in group means; significance level = 0.05):

**The missingness of `avg_rating` does depend on `n_steps`** (observed |diff| = 1.493, p ≈ 0.000). Recipes that go unrated have a systematically different number of steps than rated recipes — recipe complexity appears to play a role in whether a recipe attracts any engagement.

**The missingness of `avg_rating` does not depend on `sodium_pdv`** (observed |diff| = 0.350, p = 0.871). There is no evidence that sodium content relates to whether a recipe receives a rating.

The chart below shows the empirical distribution of the test statistic for the `n_steps` permutation test. The observed value of 1.493 sits entirely outside the simulated distribution, explaining the near-zero p-value.

<iframe src="assets/missingness_nsteps.html" width="800" height="500" frameborder="0"></iframe>

These results suggest the missingness of `avg_rating` is best described as **MAR** with respect to observed recipe features. Dropping recipes with missing `avg_rating` for modeling is unavoidable since they have no target value; the MAR dependency on observed features means this introduces only mild potential bias.

---

## Hypothesis Testing

**Question:** Do recipes with above-median calorie content receive higher average ratings than recipes with below-median calorie content?

**Null hypothesis:** The average rating of recipes with above-median calorie content is the same as that of recipes with below-median calorie content; any observed difference is due to random chance.

**Alternative hypothesis:** Recipes with above-median calorie content have higher average ratings than below-median-calorie recipes.

**Test statistic:** Difference in group means (mean rating of above-median-calorie recipes minus mean rating of below-median-calorie recipes). We chose this statistic because the alternative is directional and we are comparing a numeric outcome across two groups defined by a binarized quantitative variable.

**Significance level:** 0.05

**Results:** The observed difference was **−0.0084** (high-calorie recipes were rated marginally lower, not higher), and a permutation test with 1,000 simulations yielded a **p-value of 0.961**. Since this far exceeds 0.05, we fail to reject the null hypothesis. The data provide no evidence that higher-calorie recipes receive higher ratings. Because this is a permutation test and not a controlled experiment, we cannot conclude that calorie content has no effect on ratings — only that we found no statistically significant evidence for the hypothesized direction.

---

## Framing a Prediction Problem

**Prediction problem:** Predict the average rating (`avg_rating`) of a recipe given its characteristics at the time of publication.

**Type:** Regression — `avg_rating` is a continuous variable ranging from 1 to 5.

**Response variable:** `avg_rating`, the most direct quantitative measure of how users responded to a recipe. Predicting it lets us test whether recipe characteristics contain any signal about user satisfaction before anyone has tried the recipe.

**Evaluation metric:** RMSE (Root Mean Squared Error). RMSE is interpretable in the same units as the response variable (rating stars) and penalizes larger prediction errors more heavily than MAE. We prefer it over R² because R² is relative to the variance of the target, making it harder to interpret intuitively when that variance is low.

**Time of prediction:** All features used — `n_steps`, `minutes`, `n_ingredients`, `calories`, `protein_pdv`, `is_easy` (from tags), and `has_description` — are fully known at the moment a recipe is published, before any user rating exists. No leakage.

---

## Baseline Model

**Model:** Linear Regression in a single sklearn `Pipeline`.

**Features:** `n_steps` (quantitative) and `minutes` (quantitative). Both are left as-is — no encoding required for numeric columns. We chose these as the most direct measures of recipe effort available at time of publication.

**Performance** on a held-out 20% test set (random_state = 42):
- RMSE: **0.6359**
- R²: **−0.0002**

This is not a good model. An R² slightly below zero means it performs marginally worse than simply predicting the mean rating for every recipe — it captures essentially no variance in `avg_rating`. This is expected: step count and prep time alone carry little information about how users will rate a recipe, and the narrow rating distribution leaves very little variance to explain.

---

## Final Model

**Features added:**

| Feature | How created | Rationale |
|:---|:---|:---|
| `is_easy` | Binary: 1 if 'easy' in recipe tags | Recipes tagged 'easy' attract less experienced cooks who tend to give high ratings when a recipe succeeds, calibrating expectations differently from complex recipes. |
| `has_description` | Binary: 1 if description is not null | Contributors who write descriptions invest more care in presentation, likely proxying for recipe quality. |
| `QuantileTransformer(calories)` | In-pipeline transform | Calories has extreme right-skew; mapping to a normal distribution helps the model use calorie information across the full range. |
| `log1p(minutes)` | In-pipeline transform | Cooking time has a very long right tail; log-compression makes the feature more informative. |

We also added `n_ingredients` and `protein_pdv` to expand coverage of recipe characteristics.

**Model:** `RandomForestRegressor` — captures non-linear relationships and feature interactions that a linear model cannot, without requiring scaled inputs.

**Hyperparameter tuning:** `GridSearchCV` with 3-fold CV on the training set, searching over `n_estimators` ∈ {50, 100} and `max_depth` ∈ {5, 10, None}.

**Best parameters:** `max_depth = 5`, `n_estimators = 50`. The shallow optimal depth confirms that the signal is weak — deeper trees overfit on noise rather than learning meaningful patterns.

**Performance** (same held-out test set as baseline):
- RMSE: **0.6350** (baseline: 0.6359) ✓ improved
- R²: **0.0027** (baseline: −0.0002) ✓ improved

Both metrics improved. The R² turning positive represents a genuine improvement. The modest absolute gain reflects the data-generating process: self-selection in recipe ratings leaves little variance for any model to capture. The improvement demonstrates that recipe characteristics carry a small but real amount of predictive signal when modeled with appropriate feature engineering and non-linear methods.

---

## Fairness Analysis

**Groups:**
- **Group X:** Recipes with above-median calorie content
- **Group Y:** Recipes with below-median calorie content

**Evaluation metric:** RMSE (regression; classification metrics do not apply).

**Null hypothesis:** The model is fair — its RMSE for high-calorie and low-calorie recipes are roughly equal, and any observed difference is due to random chance.

**Alternative hypothesis:** The model performs differently (in RMSE) for high-calorie versus low-calorie recipes.

**Test statistic:** \|RMSE(Group X) − RMSE(Group Y)\|. **Significance level:** 0.05.

**Results:**
- RMSE (high-calorie): 0.6403
- RMSE (low-calorie): 0.6297
- Observed \|RMSE difference\|: **0.0106**
- **P-value: 0.531**

<iframe src="assets/fairness_permutation.html" width="800" height="500" frameborder="0"></iframe>

Since the p-value of 0.531 is well above 0.05, we fail to reject the null hypothesis. The data are consistent with the model performing similarly across calorie groups. While the model performs slightly worse on high-calorie recipes, this difference is not statistically significant, and we find no evidence of unfairness with respect to caloric content.