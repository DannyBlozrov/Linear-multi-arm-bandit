# Linear-Multi-Arm-Bandit While Avoiding Detection on fixed Budget
This is the GitHub page for the Engineering Project by Daniel Blozrov and Dean Elimelech, conducted under the guidance of Asaf Cohen. The repository contains simulations and an algorithm designed to solve the multi-armed bandit problem, based on our paper, which can be found at  [Typst](https://typst.app/project/rID3L_KmAjmQz75AsGRRrk). 


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
After installing the dependencies specified in `requirements.txt` and adjusting config.json(as explained at the bottom of the page) run:
``` py main.py ```

Enter the number of arm vectors, the dimension of each vector, the budget limit T and the number of simulations, or just press enter to skip and use default values (50,5,400,100)
### 7. Deactivate the virtual environment
```bash
deactivate
```
### Configuration File Parameters

The configuration file (`config.json`) contains several parameters to control the behavior and settings of the program. Below is a detailed explanation of each parameter:

- **`distribution_params`**: This section defines the distribution settings for generating random variables.
  - **`method`**: Specifies the type of distribution to use for generating values. Possible options include `"uniform"` ,"paper" `"normal"` , where "paper" refers to the sines and cosines as in the article by Yang and Tan.
  - **`low`**: The lower bound for the uniform distribution.
  - **`high`**: The upper bound for the uniform distribution.
  - **`mean`**: The mean value for the normal distribution.
  - **`std_dev`**: The standard deviation for the normal distribution.
  - **`theta`**: An additional distribution setting specifically for a parameter named `theta`.
    - **`method`**: The distribution method for `theta`, similar to the general `method` parameter.
    - **`mean`**: The mean value if `theta` is generated from a normal distribution.
    - **`std_dev`**: The standard deviation if `theta` follows a normal distribution.
    - **`low`**: The lower bound for `theta` when using a uniform distribution.
    - **`high`**: The upper bound for `theta` when using a uniform distribution.

- **`k`**: An integer specifying the number of arms in a multi-armed bandit problem.

- **`T`**: The total number of arm pulls allowed, an integer.

- **`d`**: The dimensionality of the arms, an integer.

- **`seed`**: A seed value for random number generation to ensure reproducibility.

- **`seed_use`**: A flag indicating whether to use the specified seed for random number generation. Possible values include `"True"` or `"False"`.

- **`output_file`**: The name of the file where the output data will be written to.
- **`noise_params`**: Parameters for the additive noise, currently only supports "normal" distribution , with "mean" and "std_dev" to be numbers.
- **`verbose`**: 'True' to enable debug printing, will be added on later though
- **`sim_num`**: Number of simulations of the experiment, higher numbers may lead to long processing times, no parallel computing as of yet, supports integers only.
- **`optimization`**: The method for doing the G-optimal, currently supports "danny" and "FW".
