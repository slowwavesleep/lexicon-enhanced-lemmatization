from lexenlem.models.common import conll
import argparse
from typing import List, NamedTuple
import numpy as np
from pathlib import Path
from collections import namedtuple, defaultdict

def load_file(filename):
    """Loads the CONLL file"""
    conll_file = conll.CoNLLFile(filename)
    data = conll_file.get(['lemma'], as_sentences=True)
    return data


def system_score(gold: List[str], system: List[str]) -> int:
    """Returns the accuracy of the system"""
    assert len(gold) == len(system), "The gold and system predictions are not of the same length."

    score = 0
    total = 0
    correct = 0

    for i in range(len(gold)):
        for j in range(len(gold[i])):
            total += 1
            if gold[i][j] == system[i][j]:
                correct += 1
    
    score = correct / total

    return score


def paired_bootstrap(gold: List[str], systems: List[NamedTuple], 
                     num_samples: int = 1000,
                     confidence: int = 95) -> None:
    """Evaluate two systems with the paired bootstrap method

    :param gold: The golden labels
    :param systems: Systems to compare
    :param num_samples: The number of random bootstrap samples
    """
    for system in systems:
        assert len(gold) == len(system.sents), \
            f"The gold ({len(gold)}) and {system.name} ({len(system.sents)}) predictions are not of the same length."

    system_scores = defaultdict(list)
    wins = defaultdict(int)

    n = len(gold)
    idx = list(range(n))
    alpha = (1 - confidence/100) / 2
    index_lo = int(alpha * (num_samples - 1))
    index_hi = num_samples - 1 - index_lo
    index_mid = int(num_samples / 2)

    # Resampling the sets for each system and comparing with the gold
    print('Resampling...')
    for system in systems:
        print(f'Resampling {system.name}...')
        for i in range(num_samples):
            idx_shuffled = np.random.choice(idx, size=n)
            gold_shuffled = [gold[i] for i in idx_shuffled]
            system_shuffled = [system.sents[i] for i in idx_shuffled]
            score = system_score(gold_shuffled, system_shuffled)
            system_scores[system.name].append(score)

    # Counting the wins of each system and raking them from the highers to lowest
    # Average accuracy is used for sorting
    final = []
    print('Ranking the systems...')
    for i_system in range(len(systems)):
        i_name = systems[i_system].name
        for j_system in range(i_system):
            j_name = systems[j_system].name
            for i, j in zip(system_scores[i_name], system_scores[j_name]):
                if i > j:
                    wins[(i_name, j_name)] += 1
                elif i < j:
                    wins[(j_name, i_name)] += 1
        scores = sorted(system_scores[i_name])
        final.append([i_name, scores[index_mid], scores[index_hi], scores[index_lo]])

    sorted_systems = sorted(final, key=lambda x: x[1], reverse=True)

    for rank, results in enumerate(sorted_systems):
        system_name, mid, hi, lo = results
        if rank < len(systems) - 1:
            lower_rank_system = sorted_systems[rank + 1][0]
            p_value = (wins[(lower_rank_system, system_name)] + 1) / (num_samples + 1)
            p_string = f'p={p_value:.3f}'
        else:
            p_value = 1
            p_string = ''
        print(f'{rank+1:2d}. {system_name:>30} {100*mid:5.2f} ±{50*(hi-lo):5.2f} ({100*lo:5.2f} .. {100*hi:5.2f}) {p_string}')
        if p_value < (1 - confidence/100):
            print('-' * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str,
                        help="Name of the CoNLL-U file with the gold data.")
    parser.add_argument("--systems", type=str,
                        help="Folder with the systems results.")
    parser.add_argument("--confidence", "-c", default=95, type=int,
                        help="X-percent confidence interval.")
    parser.add_argument("--num_samples", "-n", default=1000, type=int,
                        help="The number of random bootstrap samples.")
    parser.add_argument("--seed", "-s", default=1234, type=int,
                        help="Seed for random generator.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    gold_file = args.gold_file
    systems_dir = Path(args.systems)

    System = namedtuple('System', ['name', 'sents'])
    systems = []

    # Parse the corresponding .conllu files
    if isinstance(gold_file, str):
        filename = gold_file
        assert filename.endswith('conllu'), f"{filename} must be conllu file."
        gold_data = load_file(filename)

    for child in systems_dir.iterdir():
        if child.is_dir():
            try:
                filename = list(child.glob('*.conllu'))[0]
            except IndexError:
                raise FileNotFoundError('The folder should not be empty!')
            system_data = load_file(filename)
            system_name = child.stem.replace('_', ' ')
            systems.append(System(system_name, system_data))

    paired_bootstrap(gold_data, systems, 
        args.num_samples, args.confidence)


if __name__ == "__main__":
    main()

    