# ST2334 Helper

_Written and developed by [Prittam Ravi](https://github.com/prit3010 "Prittam Ravi") and [Kevin Chang](https://github.com/kevinchangjk "Kevin Chang Jon Kit")_

This is a python package building on top of SciPy's functions, designed and built for the purpose of simplifying computations for a specific module at the **National University of Singapore**, [ST2334](https://nusmods.com/modules/ST2334/probability-and-statistics "Probability and Statistics").

It may not be very useful for other applications.

## Installation

To install this package, simply use `pip`, with the following command.

```bash
python3 -m pip install st2334_helper
```

If not, you can also clone this repository. The module is located at `src/st2334_helper`.

Additionally, this module requires SciPy (version 1.7.1 or later), so make sure that you have SciPy installed and updated if necessary. You can perform the installation using pip, with the following command in your shell.

```bash
python3 -m pip install scipy
```

Or if you already have SciPy installed, you can update it by using the following command.

```bash
python3 -m pip install --upgrade scipy
```

## Guide

There are three main functionalities provided, grouped into three submodules as below:

1. General Usage: basic and convenient functions, but very limited
2. Confidence Intervals: construct confidence intervals for samples
3. Hypotheses Testing: conduct hypothesis test given data

To begin using the functions, import the module in a python shell. It is recommended to perform the imports as such, for greater efficiency during calculations.

```python
from st2334_helper import general as gn
from st2334_helper import confidence_intervals as ci
from st2334_helper import hypotheses_tests as ht
```

After which, you can use the functions as you please, by invoking them in the shell.

For more details on how to use, please refer to the `guide` directory. The directory contains three files:

- `functions.md`: A document detailing the specifications and instructions for every function
- `example.py`: An example of a script using the module, adapted from the creators' own attempt for ST2334 Quiz 4
- `template.py`: A template python script, for quick and easy work
