import numpy as np
 
def get_random_gene(chromosome, value):
    genes = [i for i, v in enumerate(chromosome) if value == v]
    return np.random.choice(genes)

def swap_genes(chromosome, gene1, gene2):
    chromosome[gene1], chromosome[gene2] = chromosome[gene2], chromosome[gene1]

