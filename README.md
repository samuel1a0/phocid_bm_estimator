# Body Mass estimator for phocid shaped animals

This repository contains a python script to clean, reconstruct and estimate phocid body measures based on 3D meshes.
Supported formats are .ply and .obj.

### Usage
Clone this repository and install the dependencies specified in the "requirements.txt" file. The sintax to run the script is as follow.
For a fully constumized run:
``` 
python phocid_bme.py in_path rec_path pre_trained weights_path degree precompute_n step elasticity subdivisions max_iterations
```

The explanation of all these parameters can be obtained by runin ``` python phocid_bme --help```, or simply ``` python phocid_bme -h```. Default values will be used in case they are not provided, so a basic run can be done with:
``` 
python phocid_bme in_path rec_path
```
Example data is provided so you can do a test by runing:

``` python phocid_bme --in_path example_data```

Unless specified otherwise, the reconstructions will be stored in "example_data/reconstructions" path.

## Important!
- This script is **not** aimed to get a perfect reconstruction of the animal, but an aproximation to make a body mass estimation. Due to the use of Poisson Surface Reconstruction, the resulting mesh will lack definition for areas that are not present in the provided model.
- The values of "elasticity", "degree" and "subdivisions" are related to one another. Increasing the subdivitions without increasing the degree value will result in a mantle with more resolution but at the same time less restriction (more elastic). A degree value of 0 will nullify all elasticity restrictions ( vertices will not be restricted for its neighbors ).