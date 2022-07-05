library(statmod)

fit_het_linear <- function(y,
                           mean_covar,
                           var_covar,
                           slope_covar,
                           tol = 1e-6,
                           maxit1 = 50,
                           maxit2 = 500,
                           trace = FALSE) {
    # fit y ~ N((mean_covar * mean_beta) (1 + slope_covar * slope_beta),
    # exp(var_covar * var_beta))))
    if (missing(slope_covar)) {
        # default back to remlscore
        fit <- remlscore(
            y = y,
            X = mean_covar,
            Z = var_covar,
            tol = tol, maxit = maxit2
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
        for (i in 1:maxit2) {
            slope <- 1 + slope_covar %*% slope_coef

            fit <- remlscore(
                y = y,
                X = sweep(mean_covar, MARGIN = 1, slope, "*"),
                Z = var_covar,
                tol = tol, maxit = maxit1
            )
            fitted_mu <- mean_covar %*% fit$beta
            fitted_var <- as.vector(exp(var_covar %*% fit$gamma))

            slope_fit <- lm.wfit(
                x = sweep(slope_covar, MARGIN = 1, fitted_mu, "*"),
                y = y - fitted_mu,
                w = 1 / fitted_var
            )
            # compare with previous iteration
            iter_diff <- max(abs(slope_fit$coef - slope_coef))
            if (iter_diff < tol) {
                break
            }
            slope_coef <- slope_fit$coef
            if (trace) {
                cat(paste0(
                    sprintf("Outer iter %+3s: inner iter [%+3s/%d] ", i, fit$iter, maxit1),
                    paste(round(slope_coef, 3), collapse = " "),
                    sprintf(" Error: %.3g\n", iter_diff)
                ))
            }
        }

        if (i == maxit2) {
            print(paste0(
                "Maximum number of iterations=", maxit2, " reached.",
                " May not have converged. Consider increase maxit2."
            ))
        } else {
            print(paste0(
                "Optimization converged in ", i, " iterations."
            ))
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

fit_het_linear2 <- function(y,
                            mean_covar,
                            var_covar,
                            slope_covar,
                            tol = 1e-6,
                            maxit1 = 50,
                            maxit2 = 500,
                            trace = FALSE) {
    # do not include intercept in any of the covariates
    # fit y ~ N((mean_covar * mean_beta) (1 + slope_covar * slope_beta),
    # exp(var_covar * var_beta))))
    if (missing(slope_covar)) {
        # default back to remlscore
        fit <- remlscore(
            y = y,
            X = cbind(1, mean_covar),
            Z = cbind(1, var_covar),
            tol = tol, maxit = maxit2
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
        intercept_coef <- mean(y)
        var_covar <- cbind(1, var_covar)
        for (i in 1:maxit2) {
            slope <- 1 + slope_covar %*% slope_coef

            fit <- remlscore(
                y = y - intercept_coef,
                X = sweep(mean_covar, MARGIN = 1, slope, "*"),
                Z = var_covar,
                tol = tol, maxit = maxit1
            )
            fitted_mu <- mean_covar %*% fit$beta
            fitted_var <- as.vector(exp(var_covar %*% fit$gamma))

            slope_fit <- lm.wfit(
                x = cbind(1, sweep(slope_covar, MARGIN = 1, fitted_mu, "*")),
                y = y - fitted_mu,
                w = 1 / fitted_var
            )
            new_slope_coef <- slope_fit$coef[2:length(slope_fit$coef)]
            intercept_coef <- slope_fit$coef[1]
            # compare with previous iteration
            iter_diff <- max(abs(new_slope_coef - slope_coef))
            if (iter_diff < tol) {
                break
            }
            slope_coef <- new_slope_coef
            if (trace) {
                cat(paste0(
                    sprintf("Outer iter %+3s: inner iter [%+3s/%d] ", i, fit$iter, maxit1),
                    paste(round(slope_fit$coef, 3), collapse = " "),
                    sprintf(" Error: %.3g\n", iter_diff)
                ))
            }
        }

        if (i == maxit2) {
            print(paste0(
                "Maximum number of iterations=", maxit2, " reached.",
                " May not have converged. Consider increase maxit2."
            ))
        } else {
            print(paste0(
                "Optimization converged in ", i, " iterations."
            ))
        }
        # lm.wfit does not provide vcov matrix, redo the fit using `lm`
        slope_fit <- lm(
            formula =
                (y - fitted_mu) ~
                cbind(1, sweep(slope_covar, MARGIN = 1, fitted_mu, "*")) - 1,
            weights = 1 / fitted_var
        )

        # Reference for slope_fit vcov
        # https://stats.stackexchange.com/questions/113987/lm-weights-and-the-standard-error
        slope_fit_vcov <- vcov(slope_fit) / summary(slope_fit)$sigma^2
        return(list(
            mean_beta = fit$beta,
            var_beta = fit$gamma,
            slope_beta = slope_fit$coef[2:length(slope_fit$coef)],
            intercept_beta = slope_fit$coef[1],
            mean_beta_varcov = fit$cov.beta,
            var_beta_varcov = fit$cov.gam,
            slope_beta_varcov =
                slope_fit_vcov[2:length(slope_fit$coef), 2:length(slope_fit$coef)],
            intercept_beta_varcov =
                slope_fit_vcov[c(1), c(1)]
        ))
    }
}