data {
  int N;    // Number of observations
  int y[N]; // Count at each observation
}

parameters {
  real<lower=0, upper=1> theta; // Excess zero probability
  real<lower=0> lambda;         // Poisson intensity
}

model {
  // Prior model
  theta ~ beta(1, 1);
  lambda ~ normal(0, 6.44787);

  // Observational model that mixes a Poisson with excess zeros
  for (n in 1:N) {
    real lpdf = poisson_lpmf(y[n] | lambda);
    if (y[n] == 0)
      target += log_mix(theta, 0, lpdf);
    else
      target += log(1 - theta) + lpdf;
  }
}
