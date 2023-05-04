# Koopman-based Implicit Backward Reachable Sets for Unknown Nonlinear Systems

The code accompanying our IEEE L-CSS submission that reproduces the results in the paper.

**Authors:** Haldun Balim, Antoine Aspeel, Zexiang Liu, Necmiye Ozay

## Setup 
From the base directory of this repository, install dependencies with:

~~~~
pip install -r requirements.txt
~~~~

## Usage
The following code snippets can be used to reproduce the results in the paper. They are intended to be run from the `src/` directory of this repository.

### Duffing Oscillator
To reproduce the results for the Duffing oscillator, run the following command. This script will generate the Figure 2 in the paper and it will be saved as `figures/figure2.png`:
~~~~
python main_duffing.py
~~~~

### Pendulum
To reproduce the results for the pendulum, first run the following command to collect data:
~~~~
python collect_data_pendulum.py
~~~~
Then, the following command can be used to generate the results for a given lifting function. As lifting_type `0` corresponds to the lifting with sin function, `1` corresponds to the identity lifting.
~~~~
python main_pendulum.py --lift-type {lifting_type}
~~~~
After both results are generated, the following command can be used to generate the Figure 3 in the paper and it will be saved as `figures/figure3.png`:
~~~~
python plot_pendulum.py
~~~~




