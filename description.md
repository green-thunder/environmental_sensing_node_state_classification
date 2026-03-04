1. Overview & Objective

Objective: The goal of this challenge is to predict the Target State (Label: Label_Target) of a sensing node using a dense array of anonymized topographical and engineered features.

The Task: This is a multi-class classification task. Participants must develop a model that maps 15 dense input features to one of seven distinct classification states (1–7).

Real-World Context: In advanced remote sensing and ecological monitoring, raw data often contains extreme sparsity (one-hot encoded categories) that masks the true underlying signal. To address this, the current challenge utilizes a high-novelty engineered dataset that collapses high dimensional structural and environmental data into dense, high signal features. Success requires modeling the "Environmental Synergy" of the sensing field, where attributes do not act in isolation but define specific non-linear "Niches" or states.
2. File Structure

Participants will receive the following files in the public.zip archive:

    train.csv: 464,809 observations containing the 15 input features and the target variable Label_Target.
    test.csv: 116,203 observations containing the same 15 input features and a unique Node_ID.
    sample_submission.csv: A template for submission containing the required Node_ID and a baseline "Majority Class" prediction.

3. Feature Schema

+-----------------------+------------+------------+-------------------------------------------------------+---------+
| Variable Name         | Role       | Type       | Description                                           | Units   |
+-----------------------+------------+------------+-------------------------------------------------------+---------+
| Node_ID               | ID         | Categorical| Unique sequence identifier for each observation.       | -       |
| Attr_01               | Feature    | Continuous | Primary topographical altitudinal coordinate.          | Meters  |
| Attr_02               | Feature    | Continuous | Angular orientation of the sensing node.               | Degrees |
| Attr_03               | Feature    | Continuous | Gradient coefficient of the terrain.                   | Degrees |
| Attr_04               | Feature    | Continuous | Distance to primary infrastructure access point.       | Meters  |
| Attr_05               | Feature    | Continuous | Mid-day radiation intensity index.                     | 0–255   |
| Attr_06               | Feature    | Continuous | Distance to recorded disturbance ignition point.       | Meters  |
| Group_A               | Feature    | Integer    | Categorical environmental substrate identifier.        | ID      |
| Group_B               | Feature    | Integer    | Categorical regional sector identifier.               | ID      |
| Engineered_Dist_H     | Engineered | Continuous | Calculated Euclidean path to nearest hydration source. | Meters  |
| Engineered_Flow_X     | Engineered | Continuous | Interaction index: Gradient x Hydration distance.      | Index   |
| Engineered_Density    | Engineered | Continuous | Mean proximity to infrastructure and disturbance.      | Meters  |
| Engineered_Momentum   | Engineered | Continuous | Dynamic solar shift coefficient (T1 - T3).             | Index   |
| Engineered_Stress_V   | Engineered | Continuous | Integrated altitudinal-gradient stress index.          | Index   |
| Engineered_Log_Dist   | Engineered | Continuous | Log-scaled proximity to infrastructure.               | Log(m)  |
| Label_Target          | Label      | Integer    | Ground-truth classification state (1–7).              | ID      |
+-----------------------+------------+------------+-------------------------------------------------------+---------+

4. Evaluation Metric: Macro-F1 Score

In this competition, we use the Macro-F1 Score to prevent the model from ignoring rare target states in favor of the most frequent ones. This metric treats every class as equally important, regardless of the distribution of samples in the dataset.

The Formula

The Macro-F1 is the arithmetic mean of all per-class F1 scores. For a system with NN classes (where N=7N=7):

F1macro=1N∑i=1NF1iF1macro​=N1​∑i=1N​F1i​

Where F1iF1i​ is the F1 score for a specific Target State, calculated as:

F1i=2⋅Precisioni⋅RecalliPrecisioni+RecalliF1i​=2⋅Precisioni​+Recalli​Precisioni​⋅Recalli​​

Component Definitions

To calculate the per-class F1, we use the following metrics based on True Positives (TPTP), False Positives (FPFP), and False Negatives (FNFN):

    PrecisioniPrecisioni​: TPiTPi+FPiTPi​+FPi​TPi​​ (Out of all samples predicted as State ii, how many were correct?)
    RecalliRecalli​: TPiTPi+FNiTPi​+FNi​TPi​​ (Out of all actual samples of State ii, how many did the model identify?)

5. Statistical & Physical Constraints

This dataset simulates a complex, high dimensional sensing field where success depends on navigating five core environmental hurdles:

A. Non Linear State Transitions The target labels represent discrete physical states that do not evolve linearly. Relationships between coordinates like Attr_01 and Attr_03 are governed by conditional thresholds rather than steady correlations. A signal's meaning changes entirely based on the broader environmental context.

B. Variable Signal Sensitivity The impact of proximity data is non uniform. The sensing array exhibits higher sensitivity to variance within specific ranges of the Engineered_Dist features, while other ranges show significant signal attenuation.

C. Temporal Energy Fluctuations The Engineered_Momentum feature captures the energy flux across the sensing field. Certain physical states are energetically forbidden during specific momentum extremes.

D. Heteroscedasticity & Local Volatility The dataset does not exhibit a global, uniform error distribution. Some regions of the feature space, particularly across certain Attr_01 gradients, display significantly higher local volatility.

E. Structural Redundancy & Feature Interdependence Many features, such as Attr_04 and Engineered_Density, originate from overlapping inputs, creating a dense web of multicollinearity.
6. Domain Specific Context

While the dataset provides 15 features, the underlying "Ground Truth" is governed by the structural logic of the sensing array:

A. Primary Vertical Anchors

In this sensing environment, verticality represented by features like Attr_01 and Engineered_Stress_V serves as the fundamental baseline for state distribution. Information density is not spread evenly.

B. Transmission Filtering

Variables like Engineered_Momentum and Engineered_Flow_X function as coefficients of a complex transmission process. Rather than simply adding noise, these environmental factors act as a filter that can scale or dampen the primary signal intensity.

C. Latent Categorical Hierarchies

The Group_A and Group_B identifiers contain an underlying structural hierarchy. These categories represent physical substrates with differing levels of stability and drainage. These IDs reflects their shared physical properties under high-stress conditions.

D. Asymptotic Saturation Effects

The relationship between inputs and target states often hits a Saturation Limit. Beyond certain physical thresholds such as extreme gradients in Attr_03 the probability of specific states may drop to near zero regardless of other variables, indicating a state of environmental saturation.

E. Spatial Autocorrelation & Data Leakage

The 30×3030×30 meter grid cells are not independent observations. Neighboring cells share a high degree of spatial autocorrelation. There is a high risk of neighborhood bias, where a system might learn the specific noise of a local cluster rather than the global physical laws governing the entire sensing field.
7. Submission Format

You must submit a CSV file containing your model's predictions for the test set. The file must contain exactly 116,203 rows (plus header).

Required Columns:

    Node_ID: The unique 10-character hexadecimal identifier (must match the Node_ID provided in test.csv).
    Label_Target_Predicted: Your model’s integer prediction for the classification state (1–7).

Submission Constraints:

    File Type: .csv only.
    Header: Must include the exact column names: Node_ID,Label_Target_Predicted.
    Data Format: Predictions must be discrete integers. Floating-point values (e.g., 5.01) will be cast to integers or may cause errors in the Macro-F1 calculation.
    Integrity: Every Node_ID from the test.csv must be present exactly once.

Example submission.csv:

Node_ID,Label_Target_Predicted
ce767176de,5
5ff15bccf6,5
b963fcb379,5
a1b2c3d4e5,5
