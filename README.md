# Linear-Multi-Arm-Bandit While Avoiding Detection on fixed Budget
This is the GitHub page for the Engineering Project by Daniel Blozrov and Dean Elimelech, conducted under the guidance of Asaf Cohen. The repository contains simulations and an algorithm designed to solve the multi-armed bandit problem, based on our paper, which can be found at Typst.  [Typst](https://typst.app/project/rID3L_KmAjmQz75AsGRRrk). 


# Running instructions For Simulations :
### 1. Install Python

#### On macOS:
- Python 3 is often pre-installed, but if you need to install or update it, use [Homebrew](https://brew.sh/).
  - Install Homebrew (if you don't have it):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
  - Install Python:
    ```bash
    brew install python
    ```

#### On Windows:
- Download the latest version of Python from [python.org](https://www.python.org/downloads/windows/).
- During installation, make sure to check the box **"Add Python to PATH"**.

---

### 2. Ensure `pip` is Installed

`pip` (Python's package installer) is typically installed by default. To check if it's installed and working, run the following command in your terminal (macOS) or Command Prompt/PowerShell (Windows):

```bash
pip --version
```
If it is not installed , you can run the following command to install it and make sure it is up to date
```bash
py -m ensurepip --upgrade
```
### 3. Set Up a Virtual Environment

- **Open a terminal (macOS) or Command Prompt/PowerShell (Windows)** and navigate to your project directory:
  ```bash
  cd path/to/your/project
  ```
  ```bash
  py -m venv env
  ```
---
### 4. Activate the Virtual Environment
After setting up the Virtual Envirnment :
#### On macOS:
```bash
source env/bin/activate
```
#### On Windows CMD:

  ```bash
  .\env\Scripts\activate
  ```
---
### 5. Install Dependencies
run the command
```bash
pip install -r requirements.txt
```
---
### 6. Run the simulation
After installing the dependencies specified in `requirements.txt` run:
``` py main.py ```

Enter the number of arm vectors, the dimension of each vector, the budget limit T and the number of simulations, or just press enter to skip and use default values (50,5,400,100)
### 7. Deactivate the virtual environment
```bash
deactivate
```
