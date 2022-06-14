from random import random
import calpgs
import subprocess
import os
import tempfile
import pandas as pd
import numpy as np
import random

random.seed(1234)


def test_r2diff():
    toy_data = os.path.join(calpgs.get_data_folder(), "toy.tsv")

    tmp_dir = tempfile.TemporaryDirectory()
    out_path = tmp_dir.name + "out.tsv"
    cmds = [
        "calpgs r2diff",
        f"--df {toy_data}",
        "--y y_cov",
        "--pred prs",
        "--group age,sex",
        f"--out {out_path}",
    ]
    subprocess.check_call(" ".join(cmds), shell=True)
    df_out = pd.read_csv(out_path, index_col=0, sep="\t")
    tmp_dir.cleanup()

    # test difference
    assert np.allclose(df_out["r2diff"].values, [-0.0740155, -0.0740089])
    assert np.allclose(df_out["prob>0"].values, [0.018, 0.0])


def test_model_calibrate():
    toy_data = os.path.join(calpgs.get_data_folder(), "toy.tsv")
    df = pd.read_csv(toy_data, sep="\t", index_col=0)
    calibrate_idx = np.random.choice(df.index, size=3000, replace=False)

    tmp_dir = tempfile.TemporaryDirectory()
    df_train = df.loc[calibrate_idx, :].copy()
    train_path = os.path.join(tmp_dir.name, "toy_train.tsv")
    df_train.to_csv(train_path, sep="\t", index=False)
    test_path = os.path.join(tmp_dir.name, "toy_test.tsv")
    df_test = df.loc[~df.index.isin(calibrate_idx), :].copy()
    df_test.to_csv(test_path, sep="\t", index=False)

    out_path_1 = tmp_dir.name + "model_out.tsv"
    cmds_1 = [
        "calpgs model",
        f"--df {train_path}",
        "--y y",
        "--pred prs",
        "--predstd predstd_base",
        "--ci_method scale",
        f"--out {out_path_1}",
    ]
    subprocess.check_call(" ".join(cmds_1), shell=True)

    out_path_2 = tmp_dir.name + "cali_out.tsv"
    cmds_2 = [
        "calpgs calibrate",
        f"--model {out_path_1}",
        f"--df {test_path}",
        "--pred prs",
        "--predstd predstd_base",
        f"--out {out_path_2}",
    ]
    subprocess.check_call(" ".join(cmds_2), shell=True)
    df_out = pd.read_csv(out_path_2, index_col=0, sep="\t")
    tmp_dir.cleanup()

    # test difference
    assert np.logical_and(
        df_out["cal_prs"].values > -2.22, df_out["cal_prs"].values < 2.2
    ).all()
    assert np.logical_and(
        df_out["cal_predstd"].values > 0.7, df_out["cal_predstd"].values < 0.99
    ).all()
