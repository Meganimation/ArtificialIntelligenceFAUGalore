## Prerequisites

- Python 3.7 or higher installed

### Step 1: Install Python

You may already have Python installed. Let's check.

**On Windows:**
1. Press the **Windows key** on your keyboard
2. Type `cmd` and press **Enter** — this opens the Command Prompt (a black window where you can type commands)
3. Type this and press **Enter**:
   ```
   python --version
   ```
4. If you see something like `Python 3.12.0` — you're good! Skip to Step 2.
5. If you see an error, your version is lower than Python 3.10 or it opens the Microsoft Store, you need to update or install Python:
   - Go to [python.org/downloads](https://www.python.org/downloads/)
   - Click the big yellow **"Download Python"** button
   - Run the installer
   - ⚠️ **IMPORTANT: Check the box that says "Add Python to PATH"** before clicking Install
   - Close and reopen Command Prompt, then try `python --version` again

**On Mac:**
1. Open **Terminal** (press Cmd+Space, type "Terminal", press Enter)
2. Type `python3 --version` and press **Enter**
3. If you see a version number, you're good
4. If not, install from [python.org/downloads](https://www.python.org/downloads/)

 ## Installation

 Open the folder in a CLI of your choice and in the terminal run the following:

1. (Recommended) Create a virtual environment:
   ```sh
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - On macOS/Linux:
     ```sh
     source .venv/bin/activate
     ```
   - On Windows:
     ```sh
     .venv\\Scripts\\activate
     ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

    ## Run

    - On macOS/Linux:
     ```sh
     python offlineRobot.py
     ```
   - On Windows:
     ```sh
     python offlineRobot.py
     ```
   
