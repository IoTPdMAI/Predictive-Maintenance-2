###### STAN CODE IS NOT EMBEDDED


rm(list=ls())

library(rstan)
library(parallel)
options(mc.cores = parallel::detectCores())
setwd('/Users/Rprice/Desktop/Propositions/Ratings/MI_script')

# load and drop first column
data = read.csv('subset_mi_data.csv', header = T)
data = data[,-1]

library('Matrix')

# setup data for creation of sparse matrix
data_duplicate = data
data_duplicate$reviewerID  = as.numeric(as.factor(data$reviewerID))
data_duplicate$asin   = as.numeric(as.factor(data$asin))
data_duplicate$overall = as.numeric(data$overall)
data_duplicate = data_duplicate[,-3]

# create 1429 (number of reviewers) x 900 (number of products) sparse matrix

X = sparseMatrix(i=data_duplicate$reviewerID, 
                 j=data_duplicate$asin, 
                 x=data_duplicate$overall)


# transpose it so that reviewer scores are items and products are q of interest

X = t(X)

# make another couple of copies of the data
X_NA = X
X_model <- X


# in the main dataframe, set missing to NA
X[X==0] <- NA

# create a few importanrt quantities
N <- nrow(X)# total number of obs
J <- ncol(X) # n raters
K <- 5 # thresholds

# make main matrix numeric
X_num <- matrix(as.numeric(X), nrow = nrow(X), ncol = ncol(X), byrow = T)

# create vector of prior means

m <- function(x){mean(x, na.rm = T)}
mus <- apply(X_NA, 1, m)
mus_s <- scale(mus)
mus_s <- mus_s[,1]

## IMPORTANT - GROUP CREATION
# approach 1
# analogy to nationality - latent class membership

library(poLCA)
X_lca <- X_num
X_lca[which(is.na(X_lca))] <- 0
X_lca <- X_lca + 1


X_lca <- t(X_lca) # transpose it to make ratings of items into indicators
X_lca <- X_lca[,1:50]  # take only the first 50 (this is silly but merely instructive)
X_lca <- data.frame(X_lca) 
names <- paste('Y_', 1:ncol(X_lca), sep = '') # make var names
colnames(X_lca) <- names

names_collapse <- paste(names,collapse = ',')
f1 <- cbind(names_collapse) ~1
# this is the preferred way
f2 = as.formula(paste("cbind(", paste("Y_", c(1:ncol(X_lca)), sep = "", collapse = ","), ")~1", sep ="", collapse=""))

attach(X_lca)
out <- poLCA(f2, X_lca, nclass = 5)
detach(X_lca)

# create J-length vector of classes
cdata <- out$predclass
C <- 5

# approach 2
# clustering based off of cosine similarity matrix

apply_cosine_similarity <- function(df){
  cos.sim <- function(df, ix) 
  {
    A = df[ix[1],]
    B = df[ix[2],]
    return( sum(A*B)/sqrt(sum(A^2)*sum(B^2)) )
  }   
  n <- nrow(df) 
  cmb <- expand.grid(i=1:n, j=1:n) 
  C <- matrix(apply(cmb,1,function(cmb){ cos.sim(df, cmb) }),n,n)
  C
}

X_cosine <- X_num
X_cosine[which(is.na(X_cosine))] <- 0
X_cosine <- t(X_cosine)

# generate n by n cosine similarity matrix
C <- apply_cosine_similarity(X_cosine) 

# http://www.di.fc.ul.pt/~jpn/r/spectralclustering/spectralclustering.html

make.affinity <- function(S, n.neighboors=2) {
  N <- length(S[,1])
  
  if (n.neighboors >= N) {  # fully connected
    A <- S
  } else {
    A <- matrix(rep(0,N^2), ncol=N)
    for(i in 1:N) { # for each line
      # only connect to those points with larger similarity 
      best.similarities <- sort(S[i,], decreasing=TRUE)[1:n.neighboors]
      for (s in best.similarities) {
        j <- which(S[i,] == s)
        A[i,j] <- S[i,j]
        A[j,i] <- S[i,j] # to make an undirected graph, ie, the matrix becomes symmetric
      }
    }
  }
  A  
}

A <- make.affinity(C, 3)
A[1,]

D <- diag(apply(A, 1, sum))
D[1:8, 1:8]
U <- D - A
View(U[1:70, 1:70])

"%^%" <- function(M, power)
  with(eigen(M), vectors %*% (values^power * solve(vectors)))

k   <- 10
evL <- eigen(U, symmetric=TRUE)
Z   <- evL$vectors[,(ncol(evL$vectors)-k+1):ncol(evL$vectors)]

library(stats)
km <- kmeans(Z, centers=k, nstart=5)

clusters <- km$cluster
length(clusters)


# create model matrix
X_model[X_model == 0] <- -1
X_model <- matrix(as.numeric(X_model), nrow = nrow(X_model), ncol = ncol(X_model), byrow = T)

# create data matrix list
data_list <- list(N = N, K = K, J = J, C = length(unique(clusters)), X = X_model, MU = mus_s, cdata = clusters,
                  gsigmasq = .2, gsigmasqc = .2)

help(stan)
# fit model
stan.fit <- stan(file = "MROP_instruments_informative_beta.stan",
                 data = data_list, iter = 7000, warmup = 2000, chains = 1, 
                 thin = 5, init_r = .1, verbose = TRUE, cores = 12, seed = 1234, 
                 control = list(adapt_delta = .99))

o <- summary(stan.fit, pars = 'Z')
o$summary
o$summary[which.max(o$summary[,1]),]
pairs(o$summary)

