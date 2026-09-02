set.seed(1)
M <- 1000
z500 <- list()   # sqrt(N) * theta_hat for N = 500, one vector per model
for (err in c("t10", "chisq20")) {
  for (N in c(20, 100, 500)) {
    theta_hat <- numeric(M)
    for (m in 1:M) {
      x <- rnorm(N, mean = 1, sd = 2)
      e <- if (err == "t10") rt(N, df = 10) else rchisq(N, df = 20) - 20
      y <- 1 + x + 0.2 * x^2 + e
      theta_hat[m] <- mean(y)
    }
    cat(sprintf("%-7s N = %3d  Var(theta_hat) = %.5f  Var(sqrt(N) theta_hat) = %.3f\n",
                err, N, var(theta_hat), N * var(theta_hat)))
    if (N == 500) z500[[err]] <- sqrt(N) * theta_hat
  }
}
print(rbind(mean = sapply(z500, mean), sd = sapply(z500, sd)))

png("fig_q2_hist.png", width = 1800, height = 800, res = 200)
par(mfrow = c(1, 2))
xlim <- range(unlist(z500))
xlab <- "sqrt(N) * theta_hat, N = 500"
hist(z500$t10, breaks = 40, xlim = xlim, xlab = xlab,
     main = "Original model: e ~ t(10)")
hist(z500$chisq20, breaks = 40, xlim = xlim, xlab = xlab,
     main = "New model: e = v - 20, v ~ chi-squared(20)")
dev.off()
