import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from os.path import join

def plink2_assoc_to_ldpred2(plink2_assoc_path):
    assoc = pd.read_csv(plink2_assoc_path, delim_whitespace=True)
    assert np.all(assoc["A1"] == assoc["ALT"])
    assoc = assoc[
        [
            "#CHROM",
            "ID",
            "POS",
            "ALT",
            "REF",
            "OBS_CT",
            "BETA",
            "SE",
            "T_STAT",
            "P",
        ]
    ].rename(
        columns={
            "#CHROM": "CHR",
            "ID": "SNP",
            "POS": "BP",
            "ALT": "A1",
            "REF": "A2",
            "OBS_CT": "N",
            "T_STAT": "STAT",
        }
    )
    return assoc