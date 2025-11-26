# REALM-Bench Windows Installation & Usage Guide

## 1. Introduction

This guide provides step-by-step instructions for setting up and running the `REALM-Bench` framework with the Compensation Integration on a Windows machine. This guide assumes you are starting from scratch.

---

## 2. Prerequisites

Before you begin, ensure you have the following installed on your Windows machine:

1.  **Python 3.10 or higher**:
    *   Download from [python.org](https://www.python.org/downloads/windows/).
    *   **Important**: During installation, check the box that says **"Add Python to PATH"**.
2.  **Git**:
    *   Download from [git-scm.com](https://git-scm.com/download/win).
3.  **Visual Studio Code (Recommended)**:
    *   Download from [code.visualstudio.com](https://code.visualstudio.com/).
    *   This provides a good integrated terminal.

---

## 3. Installation Steps

### Step 1: Open the Terminal
You can use **Command Prompt** (cmd) or **PowerShell**.
1.  Press `Win + R`.
2.  Type `cmd` (or `powershell`) and press Enter.

### Step 2: Clone the Repository
Navigate to the folder where you want to store the project (e.g., Documents) and clone your repository.

```cmd
cd Documents
git clone https://github.com/Kavirubc/REALM-Bench.git
cd REALM-Bench
```

### Step 3: Create a Virtual Environment
A virtual environment allows you to install libraries for this project without affecting your system-wide Python installation.

```cmd
python -m venv .venv
```

### Step 4: Activate the Virtual Environment
This step tells your terminal to use the Python version and libraries inside the `.venv` folder.

**For Command Prompt (cmd):**
```cmd
.venv\Scripts\activate
```

**For PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```
*Note: If you get a permission error in PowerShell, run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` and try again.*

You should see `(.venv)` appear at the beginning of your command line prompt.

### Step 5: Install Dependencies
Now install all the required libraries using `pip`.

```cmd
pip install -r requirements.txt
```

*Note: This may take a few minutes.*

---

## 4. Configuration (API Keys)

The system needs your Google Gemini API key to run the compensation tests.

1.  In the `REALM-Bench` folder, create a new file named `.env`.
2.  Open this file in Notepad or VS Code.
3.  Add your API keys in the following format (no spaces around the `=`):

```env
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_API_KEY=your_actual_api_key_here
```

4.  Save the file.

---

## 5. Running the Benchmark

Now you are ready to run the evaluation scripts.

### Test Integration
First, run the test script to ensure everything is set up correctly.

```cmd
python test_compensation_integration.py
```
*Expected Output: You should see several green checkmarks and "✅ All tests passed!".*

### Run Compensation Evaluation
To run the specific scenarios (CT1, CT2, CT3) designed for the compensation framework:

```cmd
python run_evaluation.py --frameworks compensation --tasks CT1,CT2,CT3
```

### Run ACID Transactional Tests
To test strict transactional integrity (all-or-nothing execution):

```cmd
python run_evaluation.py --frameworks compensation --tasks P5-ACID,P6-ACID
```

### Run Comparison with SagaLLM
To compare the compensation framework against SagaLLM on ACID tests:

```cmd
python run_evaluation.py --frameworks compensation,sagallm --tasks P5-ACID,P6-ACID
```

### Run Comparison (Advanced)
To compare the compensation framework against standard LangGraph on disruption scenarios:

```cmd
python run_evaluation.py --frameworks compensation,langgraph --tasks P4,P7,P8,P9
```

---

## 6. Troubleshooting Common Windows Issues

### Issue: "python is not recognized as an internal or external command"
*   **Cause**: Python was not added to your system PATH during installation.
*   **Fix**: Reinstall Python and ensure "Add Python to PATH" is checked, or manually add the Python path to your System Environment Variables.

### Issue: "Execution of scripts is disabled on this system" (PowerShell)
*   **Cause**: Windows security setting restricting script execution.
*   **Fix**: Run this command in PowerShell:
    ```powershell
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

### Issue: Long Path Errors
*   **Cause**: Windows has a character limit for file paths which deep nested folders might exceed.
*   **Fix**: Enable "Long Paths" in Windows via the Registry Editor or Group Policy, or move the project to a shorter path like `C:\Projects\REALM-Bench`.

### Issue: Matplotlib/Font Cache Warnings
*   **Cause**: Python trying to write cache files to a protected directory.
*   **Fix**: This is usually a warning and can be ignored. The script will use a temporary folder automatically.

---

## 7. Viewing Results

After running the evaluation, results are saved in the `evaluation_results` folder.
*   **JSON/CSV Files**: Open these in Excel or VS Code to see the data.
*   **PNG Images**: These are charts visualizing the performance. You can open them with the standard Windows Photos app.

