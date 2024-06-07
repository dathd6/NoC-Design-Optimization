from optimisers.ms_bayesian import MultiSurrogateBayesian

if __name__ == "__main__":
    optimizer = MultiSurrogateBayesian(n_cores=12, core_graph=None, mesh_2d_shape=(2, 3))
    optimizer.optimize()
