Exponentially weighted moving average using Rcpp
================
Ryan Moerer

2022-12-24

Iterative operations like for loops have quite a bit of overhead in a
language like R. Thus, it is good practice to resort to a vectorised way
of doing things whenever possible. However, sometimes loop operations
cannot be easily vectorized. Therefore, it might be better to implement
your operation in a language like C++ where loops have a lot less
overhead. You can easily do this using the R packaage `Rcpp`.

## Exponentially-Weighted Moving Average in R

Here’s an R implementation of an exponentially-weighted moving average
using a loop.

``` r
ewmaR <- function(x, alpha) {
  ewma <- numeric(length(x))
  ewma[1] <- x[1]
  for (i in 2:length(x)) {
    ewma[i] <- alpha * x[i] + (1 - alpha) * ewma[i - 1]
  }
  ewma
}

ewmaR(1:20, 0.9)
```

    ##  [1]  1.000000  1.900000  2.890000  3.889000  4.888900  5.888890  6.888889
    ##  [8]  7.888889  8.888889  9.888889 10.888889 11.888889 12.888889 13.888889
    ## [15] 14.888889 15.888889 16.888889 17.888889 18.888889 19.888889

## Exponentially-Weighted Moving Average using Rcpp

And here’s a C++ implementation of that same operation using `Rcpp`.

``` r
library(Rcpp)

cppFunction('NumericVector ewmaCPP(NumericVector x, double alpha) {
  NumericVector ewma(x.size());
  ewma[0] = x[0];
  for (int i = 1; i < x.size(); ++i) {
    ewma[i] = alpha * x[i] + (1 - alpha) * ewma[i - 1];
  }
  return ewma;
}')

ewmaCPP(1:20, 0.9)
```

    ##  [1]  1.000000  1.900000  2.890000  3.889000  4.888900  5.888890  6.888889
    ##  [8]  7.888889  8.888889  9.888889 10.888889 11.888889 12.888889 13.888889
    ## [15] 14.888889 15.888889 16.888889 17.888889 18.888889 19.888889

## Comparing performance

As one can see, the EWMA implemented in C++ is magnitudes faster than
the EWMA implemented in R.

``` r
x <- runif(1e3)

bench::mark(
  ewmaR(x, 0.9),
  ewmaCPP(x, 0.9)
)[1:6]
```

    ## # A tibble: 2 × 6
    ##   expression           min   median `itr/sec` mem_alloc `gc/sec`
    ##   <bch:expr>      <bch:tm> <bch:tm>     <dbl> <bch:byt>    <dbl>
    ## 1 ewmaR(x, 0.9)    53.71µs  55.64µs    17754.    7.86KB     2.02
    ## 2 ewmaCPP(x, 0.9)   5.41µs   6.23µs   160631.   15.53KB    16.1
