import numpy as np

class SinkhornOT:
    """Log-domain Sinkhorn algorithm using Optimal Transport

    Usage:
    ot = SinkhornOT(X, Y)          # X: (n,D) source, Y: (m,D) target
    transported = ot.transport()     # numpy array of transported samples, shape (n, D)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        # Ensure source (X) and target (Y) have the same feature dimension
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must be the same dimension")

        # Store source and target as float arrays
        self.X = X.astype(float)
        self.Y = Y.astype(float)

        # Record number of samples in each distribution
        self.n, _ = X.shape
        self.m, _ = Y.shape

        # Assign uniform probability weights to source and target
        self.a = np.ones(self.n) / self.n
        self.b = np.ones(self.m) / self.m

        # 1. Compute the cost matrix 
        self.cost_matrix = self._compute_cost_matrix(self.X, self.Y)
        # Store min and range of cost values (for normalization)
        self.cm_min = self.cost_matrix.min()
        self.cm_r   = self.cost_matrix.max() - self.cost_matrix.min()
        # Normalize costs to [0, 1] so values are on a consistent scale
        self.cost_matrix = (self.cost_matrix - self.cm_min) / (self.cm_r)

        # 2. Compute epsilon (regularization value) and kernel               
        self.epsilon, self.kernel = self._find_epsilon_and_kernel(self.cost_matrix)

        # 3. Run the Sinkhorn algorithm in log-space      
        self.alpha, self.beta = self._log_sinkhorn(
            self.cost_matrix, self.a, self.b, self.kernel, self.epsilon
        )

    @staticmethod
    def _compute_cost_matrix(X, Y):
        """
        Compute squared Euclidean distance matrix between X (n,d) and Y (m,d).
        Returns (n,m) cost matrix where entry (i,j) = ||X[i] - Y[j]||^2.
        """
        return np.sum(np.abs(X[:, None] - Y[None, :]) ** 2, axis=2)

    @staticmethod
    def _find_epsilon_and_kernel(cost_matrix):
        """
        Pick a stable epsilon for Sinkhorn and build the kernel matrix.

        Tries values in logspace(1e-3, 1e2) for epsilon. 
        Returns the first epsilon where K = exp(-C/eps) has no zeros.
        """
        eps_candidates = np.logspace(-3, 2, num=200)
        for eps in eps_candidates:
            #Kernel matrix formula
            K = np.exp(-cost_matrix / eps)
            if not np.any(K == 0):
                return eps, K
        raise RuntimeError("Could not find a stable epsilon – all kernels had zeros.")

    @staticmethod
    def _log_sinkhorn(C, a, b, K, eps, max_iters=100000, tol=1e-9):
        """
        Run Sinkhorn iterations in log-space to compute dual potentials.

        Parameters:
        C : (n, m) cost matrix
        a : (n,) source weights 
        b : (m,) target weights 
        K : (n, m) kernel 
        eps : float, entropic regularization value
        max_iters : int, maximum iterations
        tol : float, stopping tolerance for convergence

        Returns:
        alpha : (n,) dual potential for source
        beta  : (m,) dual potential for target
        """
        n, m = C.shape
        alpha = np.zeros(n)
        beta = np.zeros(m)

        for i in range(max_iters):
            alpha_prev, beta_prev = alpha.copy(), beta.copy()

            # update duals in log-space
            alpha = eps * np.log(a) - eps * np.log(K @ np.exp(beta / eps))
            beta = eps * np.log(b) - eps * np.log(K.T @ np.exp(alpha / eps))

            # convergence check every iter 
            if np.linalg.norm(alpha - alpha_prev) < tol and np.linalg.norm(beta - beta_prev) < tol:
                break
        return alpha, beta

    def _compute_transport_map(self, x):
        """
        Transport a single source point x to T(x).

        Uses the dual potentials (beta) and soft assignment P 
        to compute a barycentric projection onto Y.
        """
        # squared distances from x to every target point
        distances = np.sum(np.abs(x - self.Y) ** 2, axis=1)
        # rescale using same min–max like the global cost matrix
        distances = (distances - self.cm_min) / self.cm_r

        # compute soft assignment P(x): normalized weights over targets Y
        numerators = self.b * np.exp((self.beta - distances) / self.epsilon)
        P = numerators / np.sum(numerators)

        # weighted average of target points(barycentric transport)
        return np.sum(P[:, None] * self.Y, axis=0)  

    def transport(self):
        """Apply the barycentric transport map to all source points X.
        
        Returns:
        ndarray of shape (n, d): one transported point T(x) for each source x in X.
        """
        return np.vstack([self._compute_transport_map(x) for x in self.X])

    def __call__(self):
        return self.transport()