import calprs
import subprocess
import os
import tempfile
import pandas as pd
import numpy as np


def test_r2diff():
    toy_data = os.path.join(calprs.get_data_folder(), "toy.tsv")

    tmp_dir = tempfile.TemporaryDirectory()
    out_path = tmp_dir.name + "out.tsv"
    cmds = [
        "calprs r2diff",
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