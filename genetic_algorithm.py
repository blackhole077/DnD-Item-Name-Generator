import numpy as np
from scipy.spatial.distance import cosine
from cnn_item_generator import name_generator_function
import dataset_utils as _dutils
from tensorflow.keras.models import load_model
import model as _model

from os import path
class GeneticAlgorithm():

    def __init__(self, population_size=None, individual_length=None, num_generations=None, individual_size=None, generation_function=None, fitness_function=None):
        self.population_size = population_size
        self.individual_length = individual_length
        self.num_generations = num_generations
        self.population = None
        self.rng = np.random.default_rng()
        self.individual_size = None
        self.generation_function = generation_function
        self.fitness_function = fitness_function

    def generate_population(self, **kwargs):
        if self.population is None:
            self.population = np.zeros(shape=(self.population_size, self.individual_length), dtype=int)
            for idx in range(self.population_size):
                self.population[idx] = self.generation_function(**kwargs)

    def regenerate_population(self, **kwargs):
        if self.population is None:
            self.generate_population()
        else:
            population_remainder = self.population_size - self.population.shape[0]
            new_population = np.zeros(shape=(population_remainder, self.individual_length), dtype=int)
            for idx in range(population_remainder):
                new_population[idx] = self.generation_function(**kwargs)
        self.population = np.vstack((self.population, new_population))

    def perform_single_point_crossover(self):
        target_1, target_2 = self.fetch_random_genome_pair()
        split_index = self.rng.integers(self.individual_length)
        while split_index == 0:
            split_index = self.rng.integers(self.individual_length)
        print("Target 1: {}\nTarget 2: {}\nSplit Index: {}\n".format(target_1, target_2, split_index))
        for k in range(split_index, self.individual_length):
            self.population[target_1][k], self.population[target_2][k] = self.population[target_2][k], self.population[target_1][k]
    
    def perform_two_point_crossover(self):
        target_1, target_2 = self.fetch_random_genome_pair()
        split_index_1 = self.rng.integers(self.individual_length)
        split_index_2 = self.rng.integers(self.individual_length)
        while split_index_1 == split_index_2:
            split_index_2 = self.rng.integers(self.individual_length)
        print("Target 1: {}\nTarget 2: {}\nSplit Index_1: {}\n Split Index_2: {}".format(target_1, target_2, split_index_1, split_index_2))
        for k in range(split_index_1, split_index_2):
            self.population[target_1][k], self.population[target_2][k] = self.population[target_2][k], self.population[target_1][k]

    def fetch_random_genome_pair(self):
        target_1 = self.rng.integers(self.population_size)
        target_2 = self.rng.integers(self.population_size)
        while target_1 == target_2:
            target_2 = self.rng.integers(self.population_size)
        return target_1, target_2

    def mutate_character(self, row, index):
        # Note that in this case, individual size refers to the vocabulary size
        replacement = self.rng.integers(self.individual_size)
        self.population[row][index] = replacement

    def perform_characterwise_mutation(self, row, probability_threshold):
        new_row = []
        for _, char in enumerate(row):
            if char != 0:
                if self.rng.random() <= probability_threshold:
                    replacement = self.rng.integers(4, 37)
                    while replacement in range(5, 11):
                        replacement = self.rng.integers(4, 37)
                    new_row.append(replacement)
                else:
                    new_row.append(char)
            else:
                new_row.append(char)
        return new_row

        # for index in range(len(self.population[row])):
        #     if self.rng.random() <= probability_threshold:
        #         self.mutate_character(row, index)

    def mutate_population(self, probability_threshold):
        new_rows = []
        for row in self.population:
            new_row = self.perform_characterwise_mutation(row, probability_threshold)
            new_rows.append(new_row)
        self.population = np.append(self.population, np.array(new_rows), axis=0)

    def _print(self):
        print("Population Size: {}".format(self.population_size))
        print("Individual Length: {}".format(self.individual_length))
        print("Num Generations: {}".format(self.num_generations))
        print("Population: {}".format(self.population))
    
def create_default_genetic_algorithm():
    return GeneticAlgorithm(16, None, 10, None, None)

def acs_fitness_function(population=None, sampled_data=None):
    """
        Compute the fitness score of a population using average cosine similarity.

        Given a population and a subsample representative of the true distribution,
        compute the average cosine similarity of each gene in the population and
        return the resulting list.

        Parameters
        ----------
        population : numpy.ndarray
            A matrix of shape(population_size, individual_length)
            that corresponds to the current population of text
            sequences.
        sampled_data : numpy.ndarray
            A matrix of shape() that corresponds to a subsample
            of the original dataset (which serves as the true
            distribution).
        
        Returns
        -------
        averages : list(float)
            A list of average cosine similarity scores, with
            the index corresponding to the gene it belongs to.
    """
    
    averages = []
    for gene in population:
        average_cosine_similarity = 0.0
        print("Gene: {}".format(gene))
        for sample in sampled_data:
            # Cosine similarity is measured such that closer to zero is better.
            print("Sample: {}".format(sample))
            print("Cosine: {}".format(cosine(gene, sample)))
            average_cosine_similarity += cosine(gene, sample)
        average_cosine_similarity /= len(sampled_data)
        # Only interested in the magnitude of the vector
        averages.append(average_cosine_similarity)
    return averages

def cosine_similarity_fitness_function(population=None, sampled_data=None):
    cosine_matrix = np.zeros(shape=(len(population), len(sampled_data)))
    for row, gene in enumerate(population):
        # gene = gene[gene > 3] # Filter out the PAD, START, and END token.
        for col, sample in enumerate(sampled_data):
            cos = cosine(gene, sample)
            # sample = sample[sample > 3] # Filter out the PAD, START, and END token.
            cosine_matrix[row][col] = cos
    avg_cosine_matrix = np.average(cosine_matrix, axis=1)
    return avg_cosine_matrix

def test():
    ga = GeneticAlgorithm(5, 5)
    test = np.zeros(shape=(5,5))
    test[0] = [1,2,3,4,5]
    test[1] = [6,7,8,9,0]
    test[2] = [1,3,5,7,9]
    test[3] = [2,4,6,8,0]
    test[4] = [5,2,4,0,1]
    ga.population = test
    ga._print()
    ga.perform_single_point_crossover()
    print(ga.population)
    sampled_data = np.zeros(shape=(2,5))
    sampled_data[0] = [1,5,3,4,9]
    sampled_data[1] = [2,6,8,4,1]
    cosine_matrix = cosine_similarity_fitness_function(test, sampled_data)
    print(cosine_matrix)
    arrlinds = np.argsort(cosine_matrix)
    print(arrlinds)
    test = test[arrlinds[::]]
    print(test)

ga = GeneticAlgorithm(population_size=8,
                      individual_length=150,
                      num_generations=30,
                      individual_size=37,
                      generation_function=name_generator_function)

checkpoint_dir = './training_checkpoints'
final_model_path = path.join(checkpoint_dir, 'final_model.h5')
model = load_model(final_model_path, custom_objects={'loss': _model.loss})
# Generation step
ga.generate_population(model=model, num_instances=1)
print(ga.population)
print(_dutils.remove_characters_list(_dutils.decode_entry(ga.population)))
text = _dutils.open_file('item_names_lower.txt')
# np.random.seed(42)
encoded_input_dataset = np.array([_dutils.encode_entry(input_datum + '#' * (150 - len(input_datum))) for input_datum in text], dtype=int)
sample_indices = np.random.choice(len(encoded_input_dataset), 100, replace=False)
sample_dataset = encoded_input_dataset[sample_indices]
print("Encoded Shape: {}".format(sample_dataset.shape))
# Fitness function evaluation
for i in range(ga.num_generations):
    print("GENERATION {} OF {}".format(i, ga.num_generations))
    acs_matrix = cosine_similarity_fitness_function(ga.population, sample_dataset)
    print(acs_matrix)
    # Selection Step
    arrlinds = np.argsort(acs_matrix)[::]
    print(arrlinds)
    ga.population = np.array(ga.population)[arrlinds]
    print(_dutils.remove_characters_list(_dutils.decode_entry(ga.population)))
    # Enforce elitism, cull all but top quarter of population.
    rows_to_cut = []
    for index, idx in enumerate(arrlinds):
        if acs_matrix[idx] >= 0.35:
            rows_to_cut.append(index)
    print("Removing indices {}".format(rows_to_cut))
    ga.population = np.delete(ga.population, tuple(rows_to_cut), axis=0)
    print(_dutils.remove_characters_list(_dutils.decode_entry(ga.population)))
    cut = (ga.population_size // 4)
    if ga.population.shape[0] > cut:
        ga.population = np.delete(ga.population, np.s_[cut:], axis=0)
    print(_dutils.remove_characters_list(_dutils.decode_entry(ga.population)))
    # ga.mutate_population(probability_threshold=0.10)
    ga.population = np.unique(ga.population, axis=0)
    print(_dutils.remove_characters_list(_dutils.decode_entry(ga.population)))
    ga.regenerate_population(model=model, num_instances=1)
    print(_dutils.remove_characters_list(_dutils.decode_entry(ga.population)))

# print(parent_pool)
# print(_dutils.remove_characters_list(_dutils.decode_entry(parent_pool)))
# Mutation phase
