# cpf-diff-init
Source code and materials related to the article _Conditional particle filters with diffuse initial distributions_ [(Karppinen and Vihola, 2020)][arxiv].
The code is written in [Julia] (version 1.3.1).

## Getting started

1. [Install Julia][julia-downloads]. For compatibility it is best to use version 1.3.1.
2. Clone the project with `git clone https://github.com/skarppinen/cpf-diff-init.git`.
3. Inside the project folder `cpf-diff-init`, run `julia install_dependencies.jl`.
This script will install all Julia packages that are required by the project.

## Descriptions of the source code files in the project

The relevant source code files are found in:
<!--#### config.jl
Configuration related to the project.

#### Manifest.toml and Project.toml
Configuration files related to Julia packages used in this project. -->

#### /data/covid/
Data used in the COVID-19 stochastic SEIR example.

<!-- #### /src/config/parse_settings.jl
Configuration related to the command line interface of the scripts in /src/scripts/. -->

#### /src/julia/lib/
Functionality related to particle filtering and implementations of the methods
described in the paper.

#### /src/julia/models/
Source code for the noisy random walk, stochastic volatility, multivariate normal and SEIR models.

#### /src/julia/scripts/
Scripts for running the experiments.
The full simulation experiments are computationally intensive, and the script
`download-simulation-summaries.jl` can be used to download the (postprocessed)
simulation data visible in the article from the [data repository][data-repo].
After downloading, the scripts beginning with `analyse` can be run to reproduce the results of the article.

If needed, the individual experiments can also be run with the scripts beginning with `run`.
Type
```
julia run-*experiment-name*.jl --help
```
for usage instructions.
These scripts produce raw simulation output, which can be postprocessed with the script `postprocess-simulations.jl`.
Run `julia postprocess-simulations.jl --help` for further details.

Running the scripts in this folder will produce a new folder `output` containing the generated data.

<!-- Links -->
[julia]: https://julialang.org/
[julia-downloads]: https://julialang.org/downloads/
[data-repo]: https://nextcloud.jyu.fi/index.php/s/zjeiwDoxaegGcRe
[arxiv]: https://arxiv.org/abs/2006.14877
