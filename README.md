ssNMF project
# ssNMF - semi-supervised non-negative matrix factorization tool
`ssNMF` is a semi-supervised / constrained non-negative matrix factorization (NMF) tool.
It performs NMF, non-negative least squares, or combination of the two methods. 
It is designed to cope with biological deconvolution problems.

## Introduction 
### Problem settings
There is a set of observed samples, `data`, represented by vectors of numbers in range `[0,1]`. 
The samples are assumed to be approximately linear combinations of a reference set of samples, `atlas`. 
The main purpose is to find the weights / coefficients of these combinations. 
The atlas may be unknown, and be inferred together with the coefficients. 
In this case, this is a classic NMF problem. 
On the other hand, the atlas may be completely known in advanced, and used directly to find the coefficients. 
This case is solved with the classic NNLS (non-negative least squares) linear regression. 
In between there are other - semi-supervised - scenarios: 
only some of the atlas columns are unknown and to be inferred; 
some of columns are approximated in beforehand and are to be adjusted. 
This flexible tools solves this spectrum of problems. 

<!--![alt text](docs/img/NMF.illust.png "NMF illustration")-->
<p align="center">
    <img src="docs/img/NMF.illust.png" width="450" height="600" />
</p>

## Examples
The input `data` (csv) contains the observed samples as columns in a csv file. 
The first column must be a feature/index column, and the first line must be header/titles.

For the first case, we assume we know the `atlas` table. It should be a csv file with the same format as the `data`.
a reference table/atlas (csv), with columns representing the reference samples

## Quick start
### Installation

```bash
# Clone
git clone https://github.com/nloyfer/ssNMF.git
cd ssNMF
```

TBD

<!--### Usage examples-->
<!--Now you can generate `pat.gz` and `beta` files out of `bam` files:-->
<!--```bash-->
<!--wgbstools bam2pat Sigmoid_Colon_STL003.bam-->
<!--# output:-->
<!--# Sigmoid_Colon_STL003.pat.gz-->
<!--# Sigmoid_Colon_STL003.beta-->
<!--```-->

<!--It converts data from standard formats (e.g., bam, bed) into tailored compact yet useful and intuitive formats ([pat](docs/pat_format.md), [beta](docs/beta_format.md)).-->
<!--These can be visualized in terminal, or analyzed in different ways - subsample, merge, slice, mix, segment and more.-->
