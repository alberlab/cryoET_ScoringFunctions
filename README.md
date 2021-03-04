Code to compute quality of cluster of 3D subtomograms based on different scoring functions. In reference to paper:
### Singla, J., White, K. L., Stevens, R. C., & Alber, F. (2020). Assessment of scoring functions to rank the quality of 3D subtomogram clusters from cryo-electron tomography. bioRxiv.
[Paper](https://www.biorxiv.org/content/10.1101/2020.06.23.125823v2.article-metrics)

### Requirements:
1. Python 3
2. Numpy
3. Scipy


### How to run:
The file scoring_functions.py compute score value from set of subtomograms i.e. single cluster.

```console
python scoring_functions.py -p <particles_txtfile> -m <masks_txtfile> -o <output_file> -g <Gaussian_filter_sigma> -s <scoring_function>
```
Arguments:

-p: txt file containing paths to subtomograms in the cluster

-m: txt file containing paths to masks corresponding to particles

-o: output json file name. Default scoreValues.json

-g: gaussian filter sigma value. Default value is 0

-s: scoring function to compute. By default it computes all scoring functions.

Options for scoring funtions: SFSC, gPC, amPC, FPC, FPCmw, CCC, amCCC, cPC, oPC, OS, gNSD, cNSD, oNSD, amNSD, DSD, gMI, NMI, cMI, oMI, amMI


### Test:
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
The scoreValues.json file for the test case should generate following output:
```console
{
   "SFSC": "16.093929761606248",
   "gPC": "0.0013171162630650978",
   "amPC": "-0.0068229129113881555",
   "FPC": "0.000647349118236025",
   "FPCmw": "0.0007090374765983809",
   "CCC": "0.0012697790749313046",
   "amCCC": "-0.005472779891763771",
   "cPC": "-0.6530250739866645",
   "oPC": "0.012505335839565724",
   "OS": "0.07012576783831416",
   "gNSD": "1.9973657",
   "cNSD": "5.1596947",
   "oNSD": "0.32142648",
   "amNSD": "1.9939632",
   "DSD": "0.03239156749814856",
   "gMI": "0.00039801477849872314",
   "NMI": "1.0000653470894274",
   "cMI": "0.4597008522438511",
   "oMI": "0.042851308475981856",
   "amMI": "0.18753084696800254"
}
```

### Note:

In the paper the scoring functions gNSD, cNSD, oNSD, amNSD computed over a set of clusters are minmax normalized and subtracted from 1. But here the code only outputs the value of gSD, cSD, oSD and amSD. But the names are kept same to avoid confusion.
