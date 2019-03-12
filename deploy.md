# Deployment Information

For use on a new machine, we are trying conda.

We build a dedicated conda environment (as you should always do), once this
is activated, we run:

`conda env export > environment.yml`

Which creates the `.yml` file present in the root dir.

To recreate this environment, it suffices to have anaconda and run;

`conda env create -f environment.yml -n <whatever name you want>`
