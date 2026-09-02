# Q1.2
A <- matrix(c(5, 2, 2, 1), nrow = 2, byrow = TRUE)
B <- matrix(c(5, 2, 2, 5), nrow = 2, byrow = TRUE)
C <- solve(A) %*% B   # C = A^{-1} B
print(t(C))           # transpose of C
print(A %*% B)        # matrix product
print(A * B)          # element-wise product

# Q1.3
X <- matrix(0, 2, 2)
for (k in 1:100) {
  D_k <- matrix(c(k, k + 1, k^2, k / 2), nrow = 2, byrow = TRUE)
  X <- X + D_k
}
print(X)
