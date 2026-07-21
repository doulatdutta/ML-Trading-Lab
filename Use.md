📖 Step-by-Step Guide: How to Use the System
Here is a step-by-step walkthrough to run research, model training, and generate your MQL5 Expert Advisor.

Prerequisites
Make sure your MetaTrader 5 terminal is open and logged into your trading account (demo or live).
Look at the Market Watch window in MT5 and note down the exact symbol name (e.g. XAUUSD, XAUUSD.m, or GOLD).
In MT5, go to Tools -> Options -> Expert Advisors and make sure "Allow Algorithmic Trading" is checked.
Step 1: Set up the Local Environment
Ensure your Python virtual environment is activated and dependencies are installed:

powershell
# Activate your environment
.venv\Scripts\Activate.ps1
# Ensure you have the ML packages installed
pip install -e .[dev,ml]
Configure your local settings by copying the template file (this only needs to be done once):

powershell
cp Config/settings.example.yaml Config/settings.yaml
# Step 2: Start the Research Dashboard
Launch the FastAPI server and web dashboard:

powershell
python Scripts/run_dashboard.py
This will automatically open your default browser to the interactive dashboard at http://127.0.0.1:8000.

# Step 3: Download Historical Data
In the web dashboard, go to the Data Engine panel.
Ensure the Symbol matches your terminal (e.g. type XAUUSD or XAUUSD.m).
Click the Download M1 Data button, then the Download M3 Data button.
The backend will fetch the bars from your MT5 terminal, synchronize historical data, and save them in Config/raw.
# Step 4: Run a Baseline Backtest
Switch to the Backtest panel on the web dashboard.
Click the Run Backtest button.
This reads the downloaded data, runs the Bollinger Band + EMA Crossover strategy logic, and draws an equity curve with expectancy metrics.
You can adjust indicators parameters (like squeeze threshold, ATR target multiples) to see how baseline metrics change.
# Step 5: Train the XGBoost Machine Learning Model
Switch to the ML Training panel.
Click the Train XGBoost Model button.
The model reads all detected setups, maps current indicators/session features, and trains a classifier. You will instantly see feature importances and training performance metrics (Accuracy, Precision, Recall).
# Step 6: Run Walk-Forward & Monte Carlo Validation
Before trust-trading any parameters or filters, validate them:

Switch to the Walk-Forward panel and trigger validation to verify candidate performance on chronological out-of-sample data splits.
Go to the Monte Carlo panel and click to run 10,000 permutations to test drawdowns and ensure the edge isn't due to random luck.
# Step 7: Generate the MQL5 EA
Under the EA Generation tab, review the optimized parameters and the logical rules extracted by the machine learning decision tree.
Click Generate EA.
The python backend outputs a compile-ready, fully functional .mq5 Expert Advisor file containing your setup triggers, optimized exits, and ML session/trend filters inside MQL5/Experts/.
Drag and drop this EA onto your MT5 charts to run it in the MT5 Strategy Tester or demo trade.