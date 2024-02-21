import numpy as np

def generate_linear_bandit_instance(k=10, d=3,mean =0,sigma = 1):
    """

    :param K: number of samples of a_i..a_k
    :param d: the dimension of each sample
    :param sigma: the variance of the additive white gaussian noise
    :return:a vector of arm vectors, theta star, reward vector
    """
    # Generate random arm vectors
    arm_vectors = np.random.rand(k, d)
    # Generate random theta star
    theta = np.random.rand(d).reshape(d,1)
    # noise_vector = np.random.multivariate_normal(mean*np.ones(k),cov=sigma**2*np.eye(k))
    # reward_vector = arm_vectors @ theta + noise_vector
    return arm_vectors, theta



# Function to perform dimensionality reduction
def projection_matrix(arm_vectors):
    """

    :param arm_vectors: a matrix of K vectors in dimension d'
    :return: an orthogonal spanning base matrix B with d rows and d' columns such that A'=B^T @ A
    """
    k,d = arm_vectors.shape
    Q, R = np.linalg.qr(arm_vectors.T)
    #Q is an orthogonal matrix
    #R is an upper triangular matrix
    nonzero_indices = np.where(~np.all(R == 0, axis=1))[0]
    projection_matrix = arm_vectors[nonzero_indices] #this is B^T before normalization
    return projection_matrix


def choosing_algorithm(t,A,k):
    # = lambda t,A,k: A[np.random.randint(low = 0, high= k)]
    return A[t%k]

# Simulate fixed-budget arm pulls
def simulate_fixed_budget(k=10, d=3, T=100,mean=0,sigma=0,choosing_algorithm = choosing_algorithm):
    arm_vectors, theta = generate_linear_bandit_instance()
    V = np.zeros((d,d))
    u = np.zeros((d,1))
    for t in range(T):
        r = choosing_algorithm(t, arm_vectors, k).reshape((d, 1))
        V = V + r @ r.T
        x_t = theta.T@ r + np.random.normal(loc=mean,scale=sigma)
        u = u + np.multiply(x_t,r)
    V_inverse = np.linalg.inv(V)
    theta_estimate = V_inverse @ u
    print(f"theta = {theta}")
    print(f"theta estimate  = {theta_estimate}")






if __name__ == "__main__":
    simulate_fixed_budget()


