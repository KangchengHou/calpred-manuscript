# Compile data

1. `qsub subset_snps.sh` to subset the hapmap3 snps.
2. Form eur_train, eur_val, eur_test, admix `.fam` to partition the individuals.
3. Run this command to form PLINK files

```bash
for prefix in eur_train eur_val eur_test admix; do
    qsub subset_indiv.sh ${prefix}
done
```