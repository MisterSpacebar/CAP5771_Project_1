class Apriori:
    def __init__(self, min_support, min_confidence):
        """
        Initialize the Apriori algorithm with minimum support and confidence thresholds.

        Parameters:
        min_support (float): The minimum support threshold.
        min_confidence (float): The minimum confidence threshold.

        Output:
        None
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}  # Dictionary to store frequent itemsets with their support counts, organized by size
        self.rules = []  # List to store generated association rules

    def generate_candidates(self, prev_frequent_itemsets, k):
        """
        Generate candidate itemsets of size k from previous frequent itemsets.

        Parameters:
        prev_frequent_itemsets (set of frozensets): Frequent itemsets from the previous iteration.
        k (int): The size of itemsets to generate.

        Output:
        set of frozensets: Generated candidate itemsets.
        """
        candidates = set()

        # Get the itemsets (keys) from previous frequent itemsets
        prev_itemsets = list(prev_frequent_itemsets.keys())

        # Generate candidates by joining previous frequent itemsets
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                items1 = prev_itemsets[i]
                items2 = prev_itemsets[j]

                # For efficiency, only join if first k-2 elements are common
                # This works best when itemsets are sorted
                if k > 2:
                    items1_list = sorted(list(items1))
                    items2_list = sorted(list(items2))

                    if items1_list[:k - 2] == items2_list[:k - 2]:  # If first k-2 elements match
                        candidate = items1.union(items2)
                        if len(candidate) == k:
                            candidates.add(candidate)
                else:
                    # For k=2, simply join any two frequent 1-itemsets
                    candidate = items1.union(items2)
                    if len(candidate) == k:
                        candidates.add(candidate)

        return candidates

    def prune_candidates(self, candidates, prev_frequent_itemsets):
        """
        Prune candidates that contain non-frequent subsets.

        Parameters:
        candidates (set of frozensets): Candidate itemsets to be pruned.
        prev_frequent_itemsets (set of frozensets): Previous frequent itemsets for pruning reference.

        Output:
        set of frozensets: Pruned candidate itemsets.
        """
        pruned_candidates = set()

        for candidate in candidates:
            is_valid = True

            # Generate all k-1 size subsets of the candidate
            for item in candidate:
                subset = candidate - frozenset([item])
                # If any subset is not frequent, prune the candidate
                if subset not in prev_frequent_itemsets:
                    is_valid = False
                    break

            if is_valid:
                pruned_candidates.add(candidate)

        return pruned_candidates

    def count_support(self, transactions, candidates):
        """
        Count the support of each candidate itemset in the transactions.

        Parameters:
        transactions (list of sets): Each set represents a transaction containing items.
        candidates (set of frozensets): Candidate itemsets whose support needs to be counted.

        Output:
        dict: Dictionary mapping candidate itemsets to their support counts.
        """
        support_counts = {itemset: 0 for itemset in candidates}

        for transaction in transactions:
            for itemset in candidates:
                if itemset.issubset(transaction):
                    support_counts[itemset] += 1

        # Convert counts to support values (0-1 range)
        return {itemset: count / self.transaction_count for itemset, count in support_counts.items()}

    def eliminate_infrequent(self, candidates, candidates_support):
        """
        Eliminate candidates that do not meet the minimum support threshold.

        Parameters:
        candidates (dict): Dictionary mapping itemsets to their support counts.

        Output:
        dict: Dictionary of itemsets that meet the support threshold.
        """
        # Create a new dictionary to store frequent itemsets
        frequent_itemsets = {}

        # Iterate through each candidate and its support value
        for itemset, support in candidates_support.items():
            # Check if the support meets the minimum threshold
            if support >= self.min_support:
                # If it does, add it to the frequent itemsets
                frequent_itemsets[itemset] = support

        return frequent_itemsets

    def find_frequent_itemsets(self, transactions):
        """
        Iteratively find frequent itemsets using the Apriori algorithm steps.

        Parameters:
        transactions (list of sets): Each set represents a transaction containing items.

        Output:
        None (Updates self.frequent_itemsets)

        Example:
        transactions = [
            {"milk", "bread", "nuts", "apple"},
            {"milk", "bread", "nuts"},
            {"milk", "bread"},
            {"milk", "bread", "apple"},
            {"bread", "apple"}
        ]
        """
        self.transaction_count = len(transactions)

        # Find all unique items across all transactions
        unique_items = set()
        for transaction in transactions:
            unique_items.update(transaction)

        # Generate candidate 1-itemsets and count their support
        candidates_1 = {frozenset([item]): 0 for item in unique_items}
        for transaction in transactions:
            for itemset in candidates_1:
                if itemset.issubset(transaction):
                    candidates_1[itemset] += 1

        # Convert counts to support values
        candidates_1 = {itemset: count / self.transaction_count for itemset, count in candidates_1.items()}

        # Filter by minimum support to get frequent 1-itemsets
        self.frequent_itemsets[1] = self.eliminate_infrequent(candidates_1)

        k = 2
        while self.frequent_itemsets.get(k - 1):  # Continue until no more frequent itemsets found
            # Generate candidate itemsets
            candidates = self.generate_candidates(self.frequent_itemsets[k - 1], k)

            # Prune candidates using the Apriori property
            candidates = self.prune_candidates(candidates, self.frequent_itemsets[k - 1])

            if not candidates:
                break

            # Count support for candidates
            candidates_support = self.count_support(transactions, candidates)

            # Eliminate infrequent itemsets
            frequent_k = self.eliminate_infrequent(candidates_support)

            if frequent_k:
                self.frequent_itemsets[k] = frequent_k

            # Move to next level
            k += 1

        # Remove empty levels (though there shouldn't be any)
        self.frequent_itemsets = {k: v for k, v in self.frequent_itemsets.items() if v}

    def generate_rules(self):
        """
        Generate association rules from the discovered frequent itemsets.

        Parameters:
        None (Uses self.frequent_itemsets)

        Output:
        None (Updates self.rules)
        """
        self.rules = []

        # Only consider itemsets of size 2 or larger
        for k in self.frequent_itemsets:
            if k < 2:
                continue

            for itemset, support in self.frequent_itemsets[k].items():
                # Generate all non-empty proper subsets as potential antecedents
                for i in range(1, k):
                    for antecedent in self._get_subsets(itemset, i):
                        consequent = itemset - antecedent

                        # Calculate confidence
                        antecedent_support = self._find_support(antecedent)
                        confidence = support / antecedent_support

                        if confidence >= self.min_confidence:
                            self.rules.append((antecedent, consequent, confidence))


    def _get_subsets(self, itemset, size):
        """
        Generate all subsets of a given size from an itemset.

        Parameters:
        itemset (frozenset): The itemset to generate subsets from.
        size (int): The size of subsets to generate.

        Returns:
        list: List of subsets.
        """
        from itertools import combinations
        return [frozenset(subset) for subset in combinations(itemset, size)]


    def _find_support(self, itemset):
        """
        Find the support of an itemset from stored frequent itemsets.

        Parameters:
        itemset (frozenset): The itemset to find support for.

        Returns:
        float: Support of the itemset.
        """
        k = len(itemset)
        return self.frequent_itemsets[k][itemset] if itemset in self.frequent_itemsets[k] else 0


    def run(self, transactions):
        """
        Execute the Apriori algorithm: find frequent itemsets and generate rules.

        Parameters:
        transactions (list of sets): List of transactions containing itemsets.

        Output:
        tuple: (frequent_itemsets, rules)
        """
        # self.find_frequent_itemsets(transactions)
        # self.generate_rules()
        # return self.frequent_itemsets, self.rules

        # Convert transactions to sets if they aren't already
        transactions = [set(t) if not isinstance(t, set) else t for t in transactions]

        self.find_frequent_itemsets(transactions)
        self.generate_rules()
        return self.frequent_itemsets, self.rules


def load_transactions_from_file(filename):
    """
    Load transaction data from a file with pairs in the format:
    transaction_id item_id

    Parameters:
    filename (str): Path to the file.

    Returns:
    list: List of transaction sets.
    """
    # Dictionary to store transactions
    transaction_dict = {}

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                transaction_id, item_id = parts

                # Convert to appropriate types
                transaction_id = int(transaction_id)
                item_id = int(item_id)

                # Add item to the transaction
                if transaction_id not in transaction_dict:
                    transaction_dict[transaction_id] = set()
                transaction_dict[transaction_id].add(item_id)

    # Convert dictionary to list of sets
    transactions = list(transaction_dict.values())
    return transactions

if __name__ == "__main__":
    # Load transactions from file
    transactions = load_transactions_from_file('small.txt')

    # Print some transaction statistics
    print(f"Number of transactions: {len(transactions)}")
    print(f"Sample transactions (first 3):")
    for i, transaction in enumerate(transactions[:3]):
        print(f"  Transaction {i}: {transaction}")

    # Run Apriori with reasonable thresholds for this dataset
    min_support = 0.3  # 30% of transactions
    min_confidence = 0.7  # 70% confidence in rules

    apriori = Apriori(min_support, min_confidence)
    frequent_itemsets, rules = apriori.run(transactions)

    # Print results
    print("\nFrequent Itemsets:")
    for k, itemsets in frequent_itemsets.items():
        print(f"  {k}-itemsets: {len(itemsets)}")
        # Print a few examples
        examples = list(itemsets.items())[:3]
        for itemset, support in examples:
            print(f"    {itemset} (support: {support:.3f})")

    print("\nAssociation Rules (top 5):")
    # Sort rules by confidence
    rules.sort(key=lambda x: x[2], reverse=True)
    for antecedent, consequent, confidence in rules[:5]:
        print(f"  {antecedent} => {consequent} (confidence: {confidence:.3f})")
