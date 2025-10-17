# FitPDF: Distribution fitting tools #

[![PyPI latest release](https://img.shields.io/pypi/v/fitpdf.svg)](https://pypi.org/project/fitpdf/)
[![GitHub issues](https://img.shields.io/badge/issue_tracking-GitHub-blue.svg)](https://github.com/fjankowsk/fitpdf/issues/)
[![License - MIT](https://img.shields.io/pypi/l/fitpdf.svg)](https://github.com/fjankowsk/fitpdf/blob/master/LICENSE)

This repository contains software to fit complex distribution models to observational data. This is useful for modelling pulse-energy distributions of radio pulsars or repeating fast radio bursts (FRBs). However, the software can fit any distribution data.

## Author ##

The software is primarily developed and maintained by Fabian Jankowski. For more information, feel free to contact me via: fabian.jankowski at cnrs-orleans.fr.

## Installation ##

The easiest and recommended way to install the software is via the Python command `pip` directly from the `fitpdf` GitHub software repository. For instance, to install the master branch of the code, use the following command:  
`pip install git+https://github.com/fjankowsk/fitpdf.git@master`

This will automatically install all dependencies. Depending on your Python installation, you might want to replace `pip` with `pip3` in the above command.

The latest stable version of the code should also be available on the Python package index PyPI.

## Usage ##

```console
$ fitpdf-fit -h
usage: fitpdf-fit [-h] [--fast] [--labels name [name ...]] [--mean value] [--meanthresh value] [--model {normal,lognormal,normal_lognormal}] [--ccdf] [--log] [--nbin value] [-o]
                  [--title text]
                  files [files ...]

Fit distribution data.

positional arguments:
  files                 Names of files to process. The input files must be produced by the fluence time series option of plot-profilestack.

options:
  -h, --help            show this help message and exit
  --fast                Enable fast processing. This reduces the number of MCMC steps drastically. (default: False)
  --labels name [name ...]
                        The labels to use for each input file. (default: None)
  --mean value          The global mean fluence to divide the histograms by. (default: 1.0)
  --meanthresh value    Ignore fluence data below this mean fluence threshold, i.e. select only data where fluence / mean > meanthresh. (default: -3.0)
  --model {normal,lognormal,normal_lognormal}
                        Use the specified distribution model. (default: normal_lognormal)
  --title text          Set a custom figure title. (default: None)

Output formatting:
  --ccdf                Show the CCDF (cumulative counts) instead of the PDF (differential counts). (default: False)
  --log                 Show histograms in double logarithmic scale. (default: False)
  --nbin value          The number of histogram bins to use. (default: 50)
  -o, --output          Output plots to file rather than to screen. (default: False)
```
