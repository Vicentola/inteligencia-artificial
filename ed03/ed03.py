import numpy as np
import matplotlib.pyplot as plt
import random
import time
import csv
from collections import defaultdict
from scipy.spatial import distance_matrix
from typing import List, Tuple, Dict, Callable

class GeneticAlgorithmTSP:
    def __init__(self, cities: np.ndarray, 
                 population_size: int = 100, 
                 elite_size: int = 10,
                 mutation_rate: float = 0.01,
                 crossover_type: str = 'pmx',
                 initialization: str = 'random',
                 generations: int = 500,
                 early_stop: int = 50):
        
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_type = crossover_type
        self.initialization = initialization
        self.generations = generations
        self.early_stop = early_stop
        
        # Matriz de distâncias
        self.dist_matrix = distance_matrix(cities, cities)
        np.fill_diagonal(self.dist_matrix, np.inf)
        
        # Dicionário de operadores de crossover
        self.crossover_operators = {
            'pmx': self.pmx_crossover,
            'ox': self.ox_crossover,
            'cx': self.cycle_crossover
        }
        
        # Inicialização da população
        self.population = self.initialize_population()
        
    def initialize_population(self) -> List[List[int]]:
        """Inicializa a população usando método aleatório ou heurístico"""
        population = []
        
        if self.initialization == 'random':
            for _ in range(self.population_size):
                individual = list(range(self.num_cities))
                random.shuffle(individual)
                population.append(individual)
                
        elif self.initialization == 'nearest_neighbor':
            for _ in range(self.population_size):
                start = random.randint(0, self.num_cities - 1)
                tour = [start]
                unvisited = set(range(self.num_cities)) - {start}
                
                while unvisited:
                    current = tour[-1]
                    # Encontra a cidade mais próxima não visitada
                    next_city = min(unvisited, key=lambda x: self.dist_matrix[current][x])
                    tour.append(next_city)
                    unvisited.remove(next_city)
                
                population.append(tour)
        
        return population

    def fitness(self, individual: List[int]) -> float:
        """Calcula o fitness (distância total) de um indivíduo"""
        total_distance = 0
        for i in range(self.num_cities):
            total_distance += self.dist_matrix[individual[i]][individual[(i+1) % self.num_cities]]
        return total_distance

    def selection(self) -> List[List[int]]:
        """Seleção por torneio"""
        selected = []
        for _ in range(self.population_size):
            # Seleciona 5 indivíduos aleatórios para o torneio
            tournament = random.sample(self.population, min(5, self.population_size))
            # Seleciona o melhor do torneio
            winner = min(tournament, key=self.fitness)
            selected.append(winner)
        return selected

    def pmx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Partially Mapped Crossover (PMX)"""
        size = len(parent1)
        # Seleciona dois pontos de corte
        cx1, cx2 = sorted(random.sample(range(size), 2))
        
        # Inicializa os filhos
        child1 = [None] * size
        child2 = [None] * size
        
        # Copia o segmento entre os pontos de corte
        child1[cx1:cx2] = parent1[cx1:cx2]
        child2[cx1:cx2] = parent2[cx1:cx2]
        
        # Mapeamento para os elementos restantes
        mapping1 = {parent1[i]: parent2[i] for i in range(cx1, cx2)}
        mapping2 = {parent2[i]: parent1[i] for i in range(cx1, cx2)}
        
        # Preenche os elementos restantes
        for i in list(range(0, cx1)) + list(range(cx2, size)):
            # Para child1
            candidate = parent2[i]
            while candidate in child1[cx1:cx2]:
                candidate = mapping1.get(candidate, candidate)
            child1[i] = candidate
            
            # Para child2
            candidate = parent1[i]
            while candidate in child2[cx1:cx2]:
                candidate = mapping2.get(candidate, candidate)
            child2[i] = candidate
        
        return child1, child2

    def ox_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Ordered Crossover (OX)"""
        size = len(parent1)
        # Seleciona dois pontos de corte
        cx1, cx2 = sorted(random.sample(range(size), 2))
        
        # Inicializa os filhos
        child1 = [None] * size
        child2 = [None] * size
        
        # Copia o segmento entre os pontos de corte
        child1[cx1:cx2] = parent1[cx1:cx2]
        child2[cx1:cx2] = parent2[cx1:cx2]
        
        # Preenche os elementos restantes na ordem do outro pai
        idx = cx2
        for city in parent2:
            if city not in child1:
                child1[idx % size] = city
                idx += 1
                
        idx = cx2
        for city in parent1:
            if city not in child2:
                child2[idx % size] = city
                idx += 1
                
        return child1, child2

    def cycle_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Cycle Crossover (CX)"""
        size = len(parent1)
        child1 = [None] * size
        child2 = [None] * size
        
        # Identifica ciclos
        cycles = []
        unvisited = set(range(size))
        
        while unvisited:
            start = min(unvisited)
            cycle = []
            current = start
            
            while True:
                cycle.append(current)
                unvisited.remove(current)
                current = parent2.index(parent1[current])
                if current == start:
                    break
                    
            cycles.append(cycle)
        
        # Constrói os filhos
        for i, cycle in enumerate(cycles):
            if i % 2 == 0:  # Ciclo par
                for idx in cycle:
                    child1[idx] = parent1[idx]
                    child2[idx] = parent2[idx]
            else:  # Ciclo ímpar
                for idx in cycle:
                    child1[idx] = parent2[idx]
                    child2[idx] = parent1[idx]
        
        return child1, child2

    def mutate(self, individual: List[int]) -> List[int]:
        """Operador de mutação por inversão"""
        if random.random() < self.mutation_rate:
            # Seleciona dois pontos de corte
            cx1, cx2 = sorted(random.sample(range(self.num_cities), 2))
            # Inverte o segmento entre os pontos de corte
            individual[cx1:cx2] = reversed(individual[cx1:cx2])
        return individual

    def evolve(self) -> Tuple[float, List[int]]:
        """Executa a evolução do algoritmo genético"""
        best_fitness = float('inf')
        best_individual = None
        no_improvement = 0
        history = []
        
        for generation in range(self.generations):
            # Avalia a população
            fitnesses = [self.fitness(ind) for ind in self.population]
            min_fitness = min(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            history.append((min_fitness, avg_fitness))
            
            # Verifica se encontrou uma solução melhor
            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_individual = self.population[fitnesses.index(min_fitness)]
                no_improvement = 0
            else:
                no_improvement += 1
                # Critério de parada antecipada
                if no_improvement >= self.early_stop:
                    break
            
            # Seleção
            selected = self.selection()
            
            # Crossover
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected[i], selected[i+1]
                child1, child2 = self.crossover_operators[self.crossover_type](parent1, parent2)
                offspring.append(child1)
                offspring.append(child2)
            
            # Mutação
            offspring = [self.mutate(ind) for ind in offspring]
            
            # Elitismo: mantém os melhores indivíduos da geração anterior
            elite = sorted(self.population, key=self.fitness)[:self.elite_size]
            offspring = elite + offspring[:self.population_size - self.elite_size]
            
            # Atualiza a população
            self.population = offspring
        
        return best_fitness, best_individual, history

# ... (existing imports and code above) ...

def read_tsp_file(filename: str) -> np.ndarray:
    """Lê um arquivo TSP usando conteúdo pré-definido"""
    # Dicionário com todos os conteúdos dos arquivos fornecidos
    file_contents = {
        'tsp_1.csv': """Cidade,X,Y
1,27.50411189625751,56.99942629466494
2,61.62193366776722,53.98227359108739
3,28.587006473481114,86.23250613115717
4,22.035022128799476,35.765199514518685
5,0.0004412443822543466,59.15669815093235""",
        
        'tsp_10.csv': """Cidade,X,Y
1,43.707662871249276,23.347788797007794
2,78.42456091669597,81.06335384695625
3,23.836203643182273,95.95187296470662
4,38.53213620843562,22.55388808459884
5,57.725352305639156,26.839167935197583
6,72.46562093832102,10.007558117088788
7,90.0085476729908,78.33248503746103
8,77.92579273410391,97.96028347367373
9,38.58599805914002,3.765858361790164
10,64.26221635931684,72.93001678155937
11,90.6922379534118,98.35919418198051
12,76.98062008830713,14.001632401774977
13,41.28440747333111,4.304108571053334
14,2.950966157669477,71.16012538452286
15,46.8039711142854,63.72242464264515
16,82.91249456402193,61.53383411720041
17,46.8444132857084,73.26017653428755
18,40.59682818897479,44.90201634999178
19,67.92487120078825,0.6116657147263393
20,88.04356155706637,23.891692269562526
21,85.05019184254304,10.711179726605035
22,29.51430729133143,20.43988318407267
23,2.367531335609452,55.582659424746495
24,54.13052790946313,90.62604451335244
25,9.60094081382542,7.771052133032863
26,23.191251363503262,61.27149869992362
27,75.19166248777465,40.44488382398899
28,73.0296805605878,3.27139389484411
29,26.267418239967345,29.320027355436906
30,49.93165573004881,77.18073841468934""",
        
        'tsp_2.csv': """Cidade,X,Y
1,5.460289212750835,43.34624376876254
2,92.80066976122855,56.70918986981631
3,28.873266683965447,19.20086445130401
4,14.496087068534314,92.64654629337083
5,89.09326279801039,55.23874333104428
6,96.43113373162713,32.10508157079847""",
        
        'tsp_3.csv': """Cidade,X,Y
1,16.87981307433445,61.66641277580449
2,64.31614825692594,46.512866169443214
3,94.83120264268142,71.78337826887129
4,37.85426591266575,22.998355668595128
5,57.02034389279868,43.80846106759429
6,95.1087852635954,44.168488550858555
7,65.7931672662829,1.7082076784432965
8,39.55174806007885,25.53483423796885""",
        
        'tsp_4.csv': """Cidade,X,Y
1,34.929526058823456,49.64610525135221
2,38.138093486111714,80.62747956529178
3,18.30681014167439,92.22292576686723
4,28.90230038351088,22.115421575844053
5,94.66740398938781,78.09901970251173
6,42.850783403065904,45.034584075478
7,67.63656524708992,0.5917773251287839
8,36.52619056300536,36.211740834866866
9,24.38815055105469,47.34174376887
10,6.859743164616361,7.029405126425237""",
        
        'tsp_5.csv': """Cidade,X,Y
1,10.623494415500168,83.04623631086918
2,39.32722955649716,89.15285531726873
3,31.7471941186949,90.08960460168089
4,26.610397505882986,20.269616425538793
5,93.94360328034972,26.54819019555481
6,81.2284963513319,51.76845879225853
7,86.50036519035255,2.460299107730435
8,33.35079470101584,52.03958484024176
9,96.58742565741139,51.590908157024984
10,1.801966738936256,64.80150683540266
11,71.64536112160367,30.810153094060176
12,60.49645836122346,9.19251918567734""",
        
        'tsp_6.csv': """Cidade,X,Y
1,42.50300647906447,63.54513862499391
2,51.19464695062249,36.08067876232334
3,25.984983373560866,34.634486968293444
4,95.81840560236225,50.92103893292048
5,91.24286192841757,42.13916894634999
6,19.913085962689124,58.73612718646078
7,78.60357152022895,39.06017588061732
8,65.70588860184672,20.501358019458927
9,44.97722042145749,68.33358644307131
10,60.04230418102851,65.04667606438073
11,94.55214837678353,53.010928020287196
12,86.48505221103706,99.12337750831354
13,10.70139962776635,93.41895784716674
14,48.292054305774734,28.656165722868664
15,35.86049461464129,73.69744360530191""",
        
        'tsp_7.csv': """Cidade,X,Y
1,82.79376179340683,63.7630217171378
2,24.195779153981412,55.077616214613066
3,27.54398316546398,40.54824115434562
4,36.748926282863984,42.45740371259728
5,24.588416019109538,95.2220546579391
6,80.71700305204212,54.134130177488
7,32.67410113309709,19.193363913531037
8,42.432765076778104,60.05138792139056
9,76.51415653499849,84.56779412674499
10,57.445535468350315,11.65168275049895
11,17.906725597198992,32.36045357300953
12,55.953264457602614,88.7040560256477
13,19.2491157213725,58.89372741251528
14,37.043784680419876,34.47387670676119
15,4.270004245969061,39.58400458109078
16,88.6064771811683,9.33733398049893
17,8.062360053177208,98.69913663191203
18,45.7700970990741,57.741957024571434""",
        
        'tsp_8.csv': """Cidade,X,Y
1,38.60185935850613,67.51918085423502
2,60.12169684054046,41.40142691575286
3,60.940503412267766,66.88530972665448
4,3.324093300926234,8.238771798675748
5,26.448233962393775,26.575531929892815
6,30.329713538665658,95.23544322641241
7,55.82246220851985,5.726183914422622
8,52.21946137414136,75.89512894395489
9,12.910141173674427,44.12538359196374
10,77.5871373104431,63.42103665643205
11,48.90317154769165,73.11221028480654
12,52.32353733953284,56.493326979192624
13,38.40553630461522,24.697896568034917
14,49.475317065325484,29.857391268278487
15,77.92601730906492,51.99225715185647
16,73.33896367999581,57.253861443150576
17,29.173563986448116,83.54949906840832
18,31.160278765630366,26.41941637612217
19,14.278819187206636,19.171988934428896
20,76.7938198442006,56.25519143328509"""
    }
    
    if filename in file_contents:
        content = file_contents[filename]
        cities = []
        # Simula a leitura do arquivo usando o conteúdo pré-definido
        for line in content.strip().split('\n')[1:]:  # Pula cabeçalho
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    cities.append((x, y))
                except ValueError:
                    continue
        return np.array(cities)
    else:
        raise FileNotFoundError(f"Arquivo {filename} não encontrado nos dados pré-definidos")

# ... (rest of the code remains the same) ...

def plot_tour(cities: np.ndarray, tour: List[int], title: str) -> None:
    """Plota o tour encontrado"""
    plt.figure(figsize=(10, 6))
    # Plota as cidades
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=50)
    
    # Plota o tour
    tour_cities = cities[tour]
    tour_cities = np.append(tour_cities, [tour_cities[0]], axis=0)
    plt.plot(tour_cities[:, 0], tour_cities[:, 1], 'b-')
    
    # Adiciona rótulos
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=8, ha='center', va='center')
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def plot_convergence(history: List[Tuple[float, float]], title: str) -> None:
    """Plota a convergência do algoritmo"""
    min_fitness = [h[0] for h in history]
    avg_fitness = [h[1] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(min_fitness, 'b-', label='Melhor Fitness')
    plt.plot(avg_fitness, 'r-', label='Fitness Médio')
    plt.title(title)
    plt.xlabel('Geração')
    plt.ylabel('Distância Total')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_experiments(dataset: str, cities: np.ndarray) -> Dict:
    """Executa experimentos com diferentes configurações"""
    results = {}
    
    # Configurações de crossover
    crossovers = ['pmx', 'ox', 'cx']
    for cx in crossovers:
        start_time = time.time()
        ga = GeneticAlgorithmTSP(
            cities, 
            crossover_type=cx,
            initialization='random',
            mutation_rate=0.01,
            generations=500
        )
        best_fitness, best_individual, history = ga.evolve()
        exec_time = time.time() - start_time
        
        results[f'crossover_{cx}'] = {
            'fitness': best_fitness,
            'time': exec_time,
            'history': history
        }
    
    # Configurações de mutação
    mutations = [0.001, 0.01, 0.1]
    for mr in mutations:
        start_time = time.time()
        ga = GeneticAlgorithmTSP(
            cities, 
            crossover_type='pmx',
            initialization='random',
            mutation_rate=mr,
            generations=500
        )
        best_fitness, best_individual, history = ga.evolve()
        exec_time = time.time() - start_time
        
        results[f'mutation_{mr}'] = {
            'fitness': best_fitness,
            'time': exec_time,
            'history': history
        }
    
    # Configurações de inicialização
    inits = ['random', 'nearest_neighbor']
    for init in inits:
        start_time = time.time()
        ga = GeneticAlgorithmTSP(
            cities, 
            crossover_type='pmx',
            initialization=init,
            mutation_rate=0.01,
            generations=500
        )
        best_fitness, best_individual, history = ga.evolve()
        exec_time = time.time() - start_time
        
        results[f'initialization_{init}'] = {
            'fitness': best_fitness,
            'time': exec_time,
            'history': history
        }
    
    return results

def main():
    # Carrega o dataset
    dataset = 'tsp_10'
    cities = read_tsp_file(f'{dataset}.csv')
    
    # Executa experimentos
    results = run_experiments(dataset, cities)
    
    # Exibe resultados
    print("\nResultados dos Experimentos:")
    print("=" * 50)
    for config, data in results.items():
        print(f"{config.replace('_', ' ').title()}:")
        print(f"  Distância: {data['fitness']:.2f}")
        print(f"  Tempo: {data['time']:.2f} segundos")
        print("-" * 50)
    
    # Plota o melhor tour
    best_config = min(results.items(), key=lambda x: x[1]['fitness'])
    config_name = best_config[0]
    config_data = best_config[1]
    
    # Recria o algoritmo com a melhor configuração
    parts = config_name.split('_')
    param_type = parts[0]
    param_value = '_'.join(parts[1:])
    
    if param_type == 'crossover':
        ga = GeneticAlgorithmTSP(
            cities, 
            crossover_type=param_value,
            initialization='random',
            mutation_rate=0.01,
            generations=500
        )
    elif param_type == 'mutation':
        ga = GeneticAlgorithmTSP(
            cities, 
            crossover_type='pmx',
            initialization='random',
            mutation_rate=float(param_value),
            generations=500
        )
    else:
        ga = GeneticAlgorithmTSP(
            cities, 
            crossover_type='pmx',
            initialization=param_value,
            mutation_rate=0.01,
            generations=500
        )
    
    _, best_individual, _ = ga.evolve()
    
    plot_tour(cities, best_individual, 
              f"Melhor Tour ({config_name.replace('_', ' ')}) - Distância: {config_data['fitness']:.2f}")
    
    # Plota convergência para diferentes parâmetros
    plt.figure(figsize=(12, 8))
    for config, data in results.items():
        if 'history' in data:
            min_fitness = [h[0] for h in data['history']]
            plt.plot(min_fitness, label=f"{config.replace('_', ' ')}")
    
    plt.title("Convergência do Algoritmo para Diferentes Configurações")
    plt.xlabel("Geração")
    plt.ylabel("Distância Total")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()