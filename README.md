# StartupSuccess

This project uses machine learning to predict whether a startup will be acquired or shut down using features like funding, team size, and company type.

## Deployment
[Visit the live deployment here : startup-success.streamlit.app](https://startup-success.streamlit.app)

## Models Used
- Logistic Regression
- Random Forest (best performance: ~76% accuracy)

## Dataset
Based on Crunchbase startup investment data.

## Key Insight
Startups with more funding, strong networks, and clear milestones are more likely to be acquired.


## Run Locally

To run this application on your local machine, follow these steps:
- **Please create a venv (virtual environment) before installation**

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then the local deployment should open in localhost. Enjoy!