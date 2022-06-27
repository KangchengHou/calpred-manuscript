import dask.array as da
import pandas as pd
import dapgen
import numpy as np


def simulate_het_linear(
    X: np.ndarray, Z: np.ndarray, beta: np.ndarray, gamma: np.ndarray
):
    """
    Simulate a linear model with heteroskedasticity.

    Parameters
    ----------
    X : np.ndarray
        design matrix for mean effects
    Z : np.ndarray
        design matrix for variance effects
    beta : np.ndarray
        mean effect coefficients
    gamma : np.ndarray
        variance effect coefficients
    """
    assert X.ndim == 2
    assert Z.ndim == 2
    assert beta.ndim == 1
    assert gamma.ndim == 1
    assert X.shape[0] == Z.shape[0]
    assert X.shape[1] == beta.shape[0]
    assert Z.shape[1] == gamma.shape[0]

    return np.random.normal(loc=X @ beta, scale=np.sqrt(np.exp(Z @ gamma)))


# def simulate_quant_pheno(
#     bfile_list,
#     group_list,
#     hsq,
#     causal_prop,
#     freq_file,
#     out_prefix,
#     hermodel="gcta",
#     n_sim=10,
# ):
#     assert len(bfile_list) == len(group_list)
#     geno = []
#     df_indiv = []
#     df_snp = None

#     for bfile, group in zip(bfile_list, group_list):
#         this_geno, this_df_snp, this_df_indiv = dapgen.read_bfile(bfile, snp_chunk=1024)
#         if df_snp is None:
#             df_snp = this_df_snp
#         else:
#             assert df_snp.equals(
#                 this_df_snp
#             ), ".bim should be consistent for all bfiles"
#         geno.append(this_geno)
#         this_df_indiv["GROUP"] = group
#         df_indiv.append(this_df_indiv)

#     geno = da.concatenate(geno, axis=1)
#     df_indiv = pd.concat(df_indiv)

#     assert (
#         len(set(df_indiv.index)) == df_indiv.shape[0]
#     ), "df_indiv.index should be unique"

#     # form SNP files
#     df_freq = (
#         pd.read_csv(freq_file, delim_whitespace=True)
#         .set_index("ID")
#         .reindex(df_snp.index.values)
#     )
#     df_snp["FREQ"] = df_freq["ALT_FREQS"].values
#     assert df_snp["FREQ"].isna().any() == False

#     snp_prior_var = admix.data.calc_snp_prior_var(df_snp, hermodel)

#     n_snp, n_indiv = geno.shape
#     n_causal = int(n_snp * causal_prop)

#     sim = admix.simulate.quant_pheno_1pop(
#         geno=geno, hsq=hsq, n_causal=n_causal, n_sim=n_sim, snp_prior_var=snp_prior_var
#     )

#     df_beta = pd.DataFrame(
#         sim["beta"], columns=[f"SIM_{i}" for i in range(n_sim)], index=df_snp.index
#     )
#     df_beta = pd.concat([df_snp, df_beta], axis=1)

#     df_pheno_g = pd.DataFrame(
#         sim["pheno_g"], columns=[f"SIM_{i}" for i in range(n_sim)], index=df_indiv.index
#     )
#     df_pheno_g = pd.concat([df_indiv, df_pheno_g], axis=1)

#     df_pheno = pd.DataFrame(
#         sim["pheno"], columns=[f"SIM_{i}" for i in range(n_sim)], index=df_indiv.index
#     )
#     df_pheno = pd.concat([df_indiv, df_pheno], axis=1)

#     for suffix, df in zip(
#         ["beta", "pheno_g", "pheno"], [df_beta, df_pheno_g, df_pheno]
#     ):
#         df.to_csv(f"{out_prefix}.{suffix}.tsv", sep="\t")