# FitPDF: Distribution fitting tools #

[![PyPI latest release](https://img.shields.io/pypi/v/fitpdf.svg)](https://pypi.org/project/fitpdf/)
[![GitHub issues](https://img.shields.io/badge/issue_tracking-GitHub-blue.svg)](https://github.com/fjankowsk/fitpdf/issues/)
[![License - MIT](https://img.shields.io/pypi/l/fitpdf.svg)](https://github.com/fjankowsk/fitpdf/blob/master/LICENSE)

This repository contains software to fit complex distribution models to observational data. This is useful for modelling pulse-energy distributions of radio pulsars or repeating fast radio bursts (FRBs). However, the software can fit any distribution data.

## Author ##

The software is primarily developed and maintained by Fabian Jankowski. For more information, feel free to contact me via: fabian.jankowski at cnrs-orleans.fr.

## Paper ##

The corresponding paper is currently in preparation.

## Citation ##

If you make use of the software, please add a link to this repository and cite our corresponding paper. See above and the CITATION and CITATION.bib files.

## Installation ##

The easiest and recommended way to install the software is via the Python command `pip` directly from the `fitpdf` GitHub software repository. For instance, to install the master branch of the code, use the following command:  
`pip install git+https://github.com/fjankowsk/fitpdf.git@master`

This will automatically install all dependencies. Depending on your Python installation, you might want to replace `pip` with `pip3` in the above command.

The latest stable version of the code should also be available on the Python package index PyPI.

## Usage ##

```console
$ fitpdf-compare -h
usage: fitpdf-compare [-h] [-o] files [files ...]

Compare fits.

positional arguments:
  files         Names of files to process. The input files must be InferenceData files produced by fitpdf-fit.

options:
  -h, --help    show this help message and exit

Output formatting:
  -o, --output  Output plots to file rather than to screen. (default: False)
```

```console
$ fitpdf-fit -h
usage: fitpdf-fit [-h] [--fast] [--labels name [name ...]] [--mean value] [--meanthresh value] [--model {NL,NN,NNL}] [--ccdf] [--log] [--nbin value] [-o] [--title text] filename

Fit distribution data.

positional arguments:
  filename              Name of file to process. The input file must be produced by the fluence time series option of plot-profilestack.

options:
  -h, --help            show this help message and exit
  --fast                Enable fast processing. This reduces the number of MCMC steps drastically. (default: False)
  --labels name [name ...]
                        The labels to use for each input file. (default: None)
  --mean value          The global mean fluence to divide the histograms by. (default: 1.0)
  --meanthresh value    Ignore fluence data below this mean fluence threshold, i.e. select only data where fluence / mean > meanthresh. (default: -3.0)
  --model {NL,NN,NNL}   Use the specified distribution model, where N denotes a Normal and L a Lognormal component. For instance, the default NNL model consists of two Normal and one Lognormal distributions.
                        (default: NNL)
  --title text          Set a custom figure title. (default: None)

Output formatting:
  --ccdf                Show the CCDF (cumulative counts) instead of the PDF (differential counts). (default: False)
  --log                 Show histograms in double logarithmic scale. (default: False)
  --nbin value          The number of histogram bins to use. (default: 50)
  -o, --output          Output plots to file rather than to screen. (default: False)
```

```console
$ fitpdf-simulate -h
usage: fitpdf-simulate [-h] [--nsamp value] [-o]

Simulate distributions.

options:
  -h, --help     show this help message and exit
  --nsamp value  Number of random samples to draw from the simulated distribution. (default: 10000)

Output formatting:
  -o, --output   Output plots to file rather than to screen. (default: False)
```

## Example output ##

The images below show some example output from the program obtained when fitting simulated test data.

![Simulated distribution data](https://github.com/fjankowsk/fitpdf/raw/master/docs/simulated_data.png "Simulated distribution data")

![Example fit of the simulated test data](https://github.com/fjankowsk/fitpdf/raw/master/docs/fit_example.png "Example fit of the simulated test data")
