Business Loans Brand Ranking

1. Business context
We are a marketing aggregator operating lead-generation funnels across several verticals. This assignment covers the Business Loans (BL) vertical — financing for small businesses.
How the funnel works. A user arrives on a landing page from a paid campaign (mostly Meta). They answer a short survey about their business: entity type, credit score band, industry, how much they want to borrow, monthly revenue, time in business, and the purpose of the loan. On completion they submit contact details.
At that moment we show them a ranked list of lender brands. The user chooses one or more; each brand we successfully hand the user to pays us a fee.
Users tend to click the first brand in the list. If that brand matches the user's needs, the probability of conversion on the client's side increases. The model ranks brands according to the user's answers and some additional features for the presentation at the end of the funnel. On the other hand, brands also pay different amounts and accept different user profiles — a business with $200k monthly revenue and a 720+ credit score is valuable to some lenders and outside the criteria of others. Putting the highest-expected-value brand in the position that gets the most attention is what drives revenue per session. The expected payout is given by formula:
expected_payout = P(brand accepts this user as a lead) × payout_if_accepted
Brands are sorted descending on expected_payout. These are different questions: whether a brand takes the lead at all, and how much they pay when they do. Two different models answer these questions.

Vocabulary you will see in the data and code:
Term
Meaning
Lead
A user successfully delivered to a brand and accepted by them. This is what we get paid for.
Payout
Dollars a brand pays for one accepted lead. Roughly $1–$300 in this data.
Fund
The brand issued the loan. Downstream of us; appears in the disposition field.
Brand / client
The lender we sell the lead to (client_name). Used interchangeably.
Disposition
The brand's verdict on a lead they received.
sub1 / sub2 / sub3
Marketing attribution parameters carried through from the ad platform.
Vertical
Product category. Everything here is Business Loans.


2. The data — bl_full_data.csv
https://drive.google.com/drive/folders/1pX2IxzcJEV4Fl520ZkeyT935bNhlotY3
Sessions from 2025-12-11 to 2026-02-11.
Column groups
Identifiers and timestamps — session_id, session_dt (funnel start), conversion_dt, register_date (survey submitted). The elapsed time between start and register is engineered into a feature.
Traffic source — campaign_id, page (6 landing page variants, two dominant), sub1, sub2, sub3.
Device and geo — device_type, auto_city, auto_state, auto_country.
Survey answers — the core signal. All categorical bands rather than raw numbers:
business_type 
credit_score 
industry 
loan_amount 
monthly_revenue
time_in_business 
loan_reason

Business details — first and last names (fname, lname), cellphone, address, business_address, business_name.
Labels — client_name, payout, disposition, disposition_source, client_id. Both training targets are derived from these columns. Everything the training script needs is in this single file; there are no external lookups or joins.
3. Model architecture
Two models, multiplied:
  CatBoostClassifier  ->  P(lead)
                                    }  ->  expected_payout  ->  rank
  TabPFNRegressor     ->  $ payout
Each (user × brand) pair is scored by both models; the product determines the position of that brand in the list.
CatBoost classifier — predicts whether the brand accepts the lead. Trained on registered sessions, 25 features, native categorical handling, 800 trees, depth 8. Local, standard, fast.
TabPFN regressor — predicts the dollar payout, trained on rows where payout > 0.

Catboost
Common tree-based gradient boosting model. Resulting fitted model is saved as an artifact and used in prediction step
TabPFN
TabPFN is a pre-trained transformer for tabular data. We use it through tabpfn-client (v3.0), which is a client for a hosted API - the model runs on the provider's infrastructure and is reached over the network. Authentication is a token read from the TABPFN_TOKEN environment variable. 
It exposes a scikit-learn-style interface (fit / predict), but its execution model differs from a locally-trained estimator in ways that matter for deployment. Read the library's documentation https://docs.priorlabs.ai/quickstart#api-via-python-sdk
At the training step, an artifact with most recent training data is saved and used in the prediction step.   
4. bl_models_train.py — training
Git: https://github.com/DYK-Schemathics/Candidates/tree/dev 
Models are in branch dev
Inputs
Data
bl_full_data.csv at input_path + input_file
Authentication
TABPFN_TOKEN environment variable
Mode
train_test boolean, passed to the constructor


Two modes
train_test=True: Splits by time: the last 7 days are test, everything earlier is train. Time dependent split is applied to simulate a production run. Models are trained on the train set, then logs per-day classification reports for the classifier and MAPE/MAE for the payout regressor, including accuracy assessment with different thresholds. No artifacts are written.
train_test=False: Trains on all data and writes artifacts. No evaluation.
Outputs (production mode)
Artifact
Contents
payout_tfm_context.joblib
dict with x, y, and the ordered columns list used by the payout model
CB_bl_lead.cbm
CatBoost model, native format
all_clients.csv
Distinct client_name values — the brand universe the predictor scores against
log_file_bl_train_<timestamp>.log
Full run log


5. bl_exp_payout_predictor.py — inference
Inputs
User
Dictionary  with. In production this arrives as a dictionary; the current code reads it as a 1-row pandas DataFrame from user_data_lead.csv
Artifacts
The four files above
Auth
TABPFN_TOKEN environment variable
New user’s data
Request (dictionary). Example of a dictionary is hard-coded


Output
A dictionary keyed by brand:
{
  "fundera / nerdwallet": {"rank": 1.0, "expected_payout": 42.31},
  "businessloans.com":    {"rank": 2.0, "expected_payout": 28.05},
  "xlt":                  {"rank": 3.0, "expected_payout": 19.44},
}
This ranking is what determines the on-screen order of brands for that user. The call sits on a synchronous, user-facing path — the user is waiting on the landing page.

6. Task
Productize the two attached scripts on Databricks with MLflow. The models are given (Git link above) - do not change functions and logic. 
Data is given (Drive link above). Put it into the Databricks table to make it available to the code. Make all needed changes to make model read from the table and write artifacts, logs ans results to Databriks  
Training pipeline. Support both existing modes: train/test mode and production (trains on all data, registers artifacts). Schedule the production run weekly, Sunday 05:00. Log technical parameters to MLflow so runs are comparable. Save researcher-defined log as an artifact for accuracy assessment. Version artifacts so a serving version can be identified and rolled back.
Pay attention to tabpfn regression model - you will need to productize it in the most efficient way (https://priorlabs.ai/)
Serving. Expose the predictor as an endpoint. Input: one user's post-funnel data as a dictionary (find example in the code). Output: the ranked brand dictionary. This sits on a synchronous, user-facing path — the user is waiting on the landing page.
Load simulation. Simulate production traffic against your endpoint. Report median and tail latency Minimize latency as possible.
Push to main runnable code with full documentation in README.
