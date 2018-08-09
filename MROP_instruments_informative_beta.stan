data {
    int<lower=2> K;// categories
    int<lower=0> J; // raters
    int<lower=0> N; // N
    int<lower=0> C; // groups in pop
    int<lower=-1,upper=K> X[N,J];// data
    int<lower=1,upper=C> cdata[J]; // j group indices
    real gsigmasq;  // rater-level gamma variance around group-level gammas
    real gsigmasqc; // group-level gamma variance around population gammas
    vector[N] MU;
  }
  
  parameters {
    vector[N] Z;
    ordered[K-1] gamma[J];
    vector[K-1] gamma_mu; // population-level cutpoints
    matrix[C, (K-1)] gamma_c; // group-level cuts, rows are groups
    real<lower=0> beta[J];
  }
  model {
    vector[K] p;
    real left;
    real right;

    for(i in 1:N){
      Z[i] ~ normal(MU[i],1);
    }
    gamma_mu ~ uniform(-2, 2);

    for(c in 1:C){
      gamma_c[c] ~ normal(gamma_mu, gsigmasqc); // row-access of gamma_c
      }

    for(j in 1:J) {
      gamma[j] ~ normal(gamma_c[cdata[j]], gsigmasq);  // note row-access
      beta[j] ~ normal(1,1)T[0,];
      
      for(i in 1:N) if (X[i,j] != -1){
        left <- 0;
        for (k in 1:(K-1)){
          right <- left;
          left <- Phi_approx(gamma[j,k] - Z[i]*beta[j]);
          p[k] <- left-right;
        }
        p[K] <- 1.0 - left;
        X[i,j] ~ categorical(p);
      }
    }
  }
