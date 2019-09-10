# Open-sourced code for "A mathematical model of feeding in rodents predicts food intake and offers novel insight into feeding behaviour"

This repo contains the code and data necessary to replicate the figures and analyses in the paper. Preprints are available at biorxiv and arxiv [ADD LINKS].

## How to use
### To replicate our analysis:
- First, build the Anaconda environment: `conda env create -f environment.yml`.
- To carry out the inference, run `bash run_all.sh`.
- To recreate the figures, run the relevant Jupyter notebook.

## To carry out your own analysis:
- Move the raw CLAMS bout data into the folder `data_to_process`, with filenames in the format `YYYY-MM-DD.BX.CSV` (case sensitive), where X is the group/cage ID (typically found on line 5 of the data file).
- Create a `coding.csv` file for your dataset, following the format of the example `coding.csv` file used for our dataset.
- Run the Jupyter notebook `Preprocess.ipynb`, adjusting the rejection thresholds as necessary depending on your dataset size.
- Run the Monte Carlo inference: `bash run_all.sh`. This will generate diagnostic plots to ensure convergence, as well as trace files containing the results. These can reach several gigabytes in size for large datasets, and are pickled at the end of the MCMC program.
- Use the trace files as necessary for your analyses. You may find some of the functions in `helpers.py` useful in interacting with the traces - examples of how to use these are found in `FigurePrelims.ipynb`.

## Important files
- `PDMP_LL.py`: likelihood functions for the model. This is a good place to start if you're interested in extending/modifying the model.
- `fwd_sample.py`: forward sampling code for generating synthetic data. Useful for _in silico_ experimentation.
- `run_on_dataset.py`: PyMC3 code to carry out the inference. Contains details of the hierarchical model.

# Contact Information
If you have any questions, or are interested in extending our model or applying it to new datasets, please contact the first author (Tom McGrath) at tmm13@ic.ac.uk
