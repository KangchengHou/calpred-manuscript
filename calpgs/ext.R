library(statmod)

fit_het_linear <- function(y,
                           mean_covar,
                           var_covar,
                           slope_covar,
                           tol = 1e-6,
                           maxit = 500) {
    # fit y ~ N((mean_covar * mean_beta) (1 + slope_covar * slope_beta),
    # exp(var_covar * var_beta))))
    if (missing(slope_covar)) {
        # default back to remlscore
        fit <- remlscore(
            y = y,
            X = mean_covar,
            Z = var_covar,
            tol = tol, maxit = maxit
        )
        return(list(
            mean_beta = fit$beta,
            var_beta = fit$gamma,
            mean_beta_varcov = fit$cov.beta,
            var_beta_varcov = fit$cov.gam
        ))
    } else {
        # coordinate descent to optimize slope_beta
        slope_coef <- rep(0, ncol(slope_covar))
        for (i in 1:maxit) {
            slope <- 1 + slope_covar %*% slope_coef

            fit <- remlscore(
                y = y,
                X = sweep(mean_covar, MARGIN = 1, slope, "*"),
                Z = var_covar,
                tol = tol, maxit = maxit
            )
            fitted_mu <- mean_covar %*% fit$beta
            fitted_var <- as.vector(exp(var_covar %*% fit$gamma))

            slope_fit <- lm.wfit(
                x = sweep(slope_covar, MARGIN = 1, fitted_mu, "*"),
                y = y - fitted_mu,
                w = 1 / fitted_var
            )
            # compare with previous iteration
            if (max(abs(slope_fit$coef - slope_coef)) < tol) {
                break
            }
            slope_coef <- slope_fit$coef
            if (i == maxit) {
                print(paste0(
                    "Maximum number of iterations reached.",
                    " May not have converged. Consider increase maxit."
                ))
            }
        }
        # lm.wfit does not provide vcov matrix, redo the fit using `lm`
        slope_fit <- lm(
            formula =
                (y - fitted_mu) ~
                sweep(slope_covar, MARGIN = 1, fitted_mu, "*") - 1,
            weights = 1 / fitted_var
        )

        # Reference for slope_fit vcov
        # https://stats.stackexchange.com/questions/113987/lm-weights-and-the-standard-error
        return(list(
            mean_beta = fit$beta,
            var_beta = fit$gamma,
            slope_beta = slope_fit$coef,
            mean_beta_varcov = fit$cov.beta,
            var_beta_varcov = fit$cov.gam,
            slope_beta_varcov = vcov(slope_fit) / summary(slope_fit)$sigma^2
        ))
    }
}