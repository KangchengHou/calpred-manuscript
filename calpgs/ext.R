library(statmod)

calc_nll <- function(y,
                     mean_covar,
                     var_covar,
                     slope_covar,
                     mean_coef,
                     var_coef,
                     slope_coef,
                     intercept_covar = NULL,
                     intercept_coef = NULL) {
    # Calculate the negative log-likelihood of the data
    # given the current estimates
    if (is.null(intercept_covar)) {
        stopifnot(is.null(intercept_coef))
    }
    fitted_mean <- mean_covar %*% mean_coef
    fitted_var <- as.vector(exp(var_covar %*% var_coef))
    fitted_mean <- fitted_mean * (1 + slope_covar %*% slope_coef)
    if (!is.null(intercept_covar)) {
        fitted_mean <- fitted_mean + intercept_covar %*% intercept_coef
    }
    loglik <- dnorm(y, mean = fitted_mean, sd = sqrt(fitted_var), log = TRUE)
    return((-1) * sum(loglik))
}
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
            nll <- calc_nll(
                y = y,
                mean_covar = mean_covar,
                var_covar = var_covar,
                slope_covar = slope_covar,
                mean_coef = fit$beta,
                var_coef = fit$gamma,
                slope_coef = slope_coef,
            )
            if (trace) {
                cat(paste0(
                    sprintf(
                        "Outer iter %+3s: inner iter [%+3s/%d] ",
                        i, fit$iter, maxit1
                    ),
                    paste(round(slope_coef, 3), collapse = " "),
                    sprintf(" Error: %.3g", iter_diff),
                    sprintf(" NLL: %.2f\n", nll)
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

fit_het_linear_intercept <- function(y,
                                     mean_covar,
                                     var_covar,
                                     slope_covar,
                                     intercept_covar,
                                     tol = 1e-6,
                                     maxit1 = 50,
                                     maxit2 = 500,
                                     trace = FALSE) {
    # fit y ~ N((mean_covar * mean_beta) (1 + slope_covar * slope_beta),
    # exp(var_covar * var_beta))))
    # coordinate descent to optimize slope_beta
    n_indiv <- nrow(mean_covar)
    stopifnot(all(c(
        nrow(mean_covar), nrow(var_covar),
        nrow(slope_covar), nrow(intercept_covar)
    ) == n_indiv[1]))
    n_slope_covar <- ncol(slope_covar)
    n_intercept_covar <- ncol(intercept_covar)
    slope_coef <- rep(0, n_slope_covar)
    intercept_coef <- rep(0, n_intercept_covar)
    for (i in 1:maxit2) {
        slope <- 1 + slope_covar %*% slope_coef
        intercept <- intercept_covar %*% intercept_coef
        fit <- remlscore(
            y = y - intercept,
            X = sweep(mean_covar, MARGIN = 1, slope, "*"),
            Z = var_covar,
            tol = tol, maxit = maxit1
        )
        fitted_mu <- mean_covar %*% fit$beta
        fitted_var <- as.vector(exp(var_covar %*% fit$gamma))

        intercept_slope_fit <- lm.wfit(
            x = cbind(
                intercept_covar,
                sweep(slope_covar, MARGIN = 1, fitted_mu, "*")
            ),
            y = y - fitted_mu,
            w = 1 / fitted_var
        )
        # compare with previous iteration
        iter_diff <- max(abs(
            intercept_slope_fit$coef - c(intercept_coef, slope_coef)
        ))
        if (iter_diff < tol) {
            break
        }
        intercept_coef <- intercept_slope_fit$coef[1:n_intercept_covar]
        slope_coef <- intercept_slope_fit$coef[
            (n_intercept_covar + 1):(n_intercept_covar + n_slope_covar)
        ]
        nll <- calc_nll(
            y = y,
            mean_covar = mean_covar,
            var_covar = var_covar,
            slope_covar = slope_covar,
            mean_coef = fit$beta,
            var_coef = fit$gamma,
            slope_coef = slope_coef,
        )

        if (trace) {
            cat(paste0(
                sprintf(
                    "Outer iter %+3s: inner iter [%+3s/%d] ",
                    i, fit$iter, maxit1
                ),
                paste(round(intercept_slope_fit$coef, 3), collapse = " "),
                sprintf(" Error: %.3g", iter_diff),
                sprintf(" NLL: %.2f\n", nll)
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
    intercept_slope_fit <- lm(
        formula =
            (y - fitted_mu) ~
            cbind(
                intercept_covar,
                sweep(slope_covar, MARGIN = 1, fitted_mu, "*")
            ) - 1,
        weights = 1 / fitted_var
    )
    intercept_slope_vcov <- vcov(intercept_slope_fit) /
        summary(intercept_slope_fit)$sigma^2

    # Reference for slope_fit vcov
    # https://stats.stackexchange.com/questions/113987/lm-weights-and-the-standard-error
    return(list(
        mean_beta = fit$beta,
        var_beta = fit$gamma,
        slope_beta = intercept_slope_fit$coef[
            (n_intercept_covar + 1):(n_intercept_covar + n_slope_covar)
        ],
        intercept_beta = intercept_slope_fit$coef[1:n_intercept_covar],
        mean_beta_varcov = fit$cov.beta,
        var_beta_varcov = fit$cov.gam,
        slope_beta_varcov = intercept_slope_vcov[
            (n_intercept_covar + 1):(n_intercept_covar + n_slope_covar),
            (n_intercept_covar + 1):(n_intercept_covar + n_slope_covar),
            drop = FALSE
        ],
        intercept_beta_varcov = intercept_slope_vcov[
            1:n_intercept_covar,
            1:n_intercept_covar,
            drop = FALSE
        ]
    ))
}