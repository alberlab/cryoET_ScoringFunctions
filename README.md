Code to compute quality of cluster of 3D subtomograms based on different scoring functions. In reference to paper:
##### Singla, J., White, K. L., Stevens, R. C., & Alber, F. (2020). Assessment of scoring functions to rank the quality of 3D subtomogram clusters from cryo-electron tomography. bioRxiv.


#### Requirements:
1. Python 3
2. Numpy
3. Scipy


#### How to run:
The file scoring_functions.py compute score value from set of subtomograms i.e. single cluster.
Just download the scoring_functions.py code and run it directly in terminal as follows:

```console
python scoring_functions.py -p <particles_txtfile> -m <masks_txtfile> -g <Gaussian_filter_sigma> -s <scoring_function>
```
Arguments:

-p: txt file containing paths to subtomograms in the cluster

-m: txt file containing paths to masks corresponding to particles

-g: gaussian filter sigma value. Default value is 0

-s: scoring function to compute. By default it computes all scoring functions.

Options for scoring funtions: [SFSC, gPC, amPC, FPC, FPCmw, CCC, amCCC, cPC, oPC, OS, gNSD, cNSD, oNSD, amNSD, DSD, gMI, NMI, cMI, oMI, amMI]


#### Test:
```console
python scoring_functions.py -p ./test/particles.txt -m ./test/masks.txt
```
This command computes all scoring functions and without any gaussian filtering

Terminal Output
```console
Input particles file is ./test/particles.txt
Input masks file is ./test/masks.txt
Output file is scoreValues.json
Gaussian filter = 0
Scoring function = all

Computing Scores ...

Computing SFSC
Number of subtomograms loaded: 4

Computing gPC, amPC, FPC, FPCmw, CCC, amCCC, cPC, oPC, OS, gNSD, cNSD, oNSD, amNSD, DSD, gMI, NMI, cMI, oMI, amMI
Num of pairs:  10
Computation complete. Scores saved in scoreValues.json
```
