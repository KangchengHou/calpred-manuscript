import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from tqdm import tqdm
from typing import List


def plink2_assoc_to_ldpred2(plink2_assoc_path):
    assoc = pd.read_csv(plink2_assoc_path, delim_whitespace=True)
    assert np.all(assoc["A1"] == assoc["ALT"])
    #     flip_pos = assoc["A1"] != assoc["ALT"]
    #     print(
    #         f"{np.mean(flip_pos):.2g} of SNPs are flipped,"
    #         " fliping BETA and T_STAT at these location"
    #     )
    #     assoc.loc[flip_pos, ["BETA", "T_STAT"]] *= -1
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


def load_sim_data(
    config: str,
    DATA_DIR: str = "/u/project/sgss/UKBB/PRS-RESEARCH/02-yi-simulate-prs/subcontinenal/",
    n_sim: int = 5,
) -> List[pd.DataFrame]:
    """
    Parameters
    ----------
    config: str
        simulation parameter configuration

    DATA_DIR: str
        root directory to read the data

    n_sim: int
        number of simulations to read

    Returns
    -------
    A list of `n_sim` pd.DataFrame, with each data frame containing
        - PRS_MEAN: PRS point estimate
        - PRS_STD: PRS uncertainty standard deviation
        - PHENO: phenotype
        - PHENO_G: genetic component of phenotype
    """
    df_pheno_g = pd.read_csv(
        join(DATA_DIR, config, "data/sim.pheno_g.tsv.gz"), sep="\t"
    )
    df_pheno = pd.read_csv(join(DATA_DIR, config, "data/sim.pheno.tsv.gz"), sep="\t")

    for df in [df_pheno_g, df_pheno]:
        df.index = df.FID.astype(str) + "_" + df.IID.astype(str)
        df.drop(columns=["FID", "IID"], inplace=True)

    df_res_list = []
    for i in tqdm(range(n_sim)):
        df_prs = pd.read_csv(
            join(DATA_DIR, config, f"predict/SIM_{i}.auto.test.test_prs.tsv.gz"),
            sep="\t",
            index_col=0,
        ).drop(columns=["MEAN"])
        # center with column mean
        df_prs -= df_prs.mean(axis=0)
        df_res = pd.DataFrame(
            {"PRS_MEAN": df_prs.mean(axis=1), "PRS_STD": df_prs.std(axis=1)}
        )
        df_res["PHENO"] = df_pheno[f"SIM_{i}"].reindex(df_res.index)
        df_res["PHENO_G"] = df_pheno_g[f"SIM_{i}"].reindex(df_res.index)
        df_res_list.append(df_res)

    return df_res_list