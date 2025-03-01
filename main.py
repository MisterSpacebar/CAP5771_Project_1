import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

class Apriori:
    def __init__(self, min_support, min_confidence):
        """
        Initialize the Apriori algorithm with minimum support and confidence thresholds.

        Parameters:
        min_support (int): The minimum support threshold as absolute count.
        min_confidence (float): The minimum confidence threshold (0-1).
        """
        self.min_support = min_support  # Absolute count, not percentage
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}  # Dictionary to store frequent itemsets with their support counts
        self.rules = []  # List to store generated association rules
        self.transaction_count = 0  # Total number of transactions

        # Timing information
        self.execution_times = {
            'total': 0,
            'frequent_itemsets': 0,
            'candidate_generation': 0,
            'support_counting': 0,
            'rule_generation': 0,
            'phases': []  # Will store detailed timing for each phase/iteration
        }

    def visualize_frequent_itemsets(self, top_n=20):
        """
        Visualize the frequent itemsets discovered by the algorithm.

        Parameters:
        top_n (int): Number of top itemsets to display per level

        Returns:
        plotly.graph_objects.Figure: Interactive visualization of frequent itemsets
        """
        # Prepare data for visualization
        data = []

        for k, itemsets in self.frequent_itemsets.items():
            # Sort itemsets by support
            sorted_itemsets = sorted(itemsets.items(), key=lambda x: x[1], reverse=True)

            # Take top N itemsets for this level
            for itemset, count in sorted_itemsets[:top_n]:
                # Convert frozenset to string representation
                itemset_str = ', '.join(map(str, itemset))
                support_pct = count / self.transaction_count * 100

                data.append({
                    'Itemset': itemset_str,
                    'Size': k,
                    'Support Count': count,
                    'Support %': support_pct
                })

        # Create DataFrame
        df = pd.DataFrame(data)

        if len(df) == 0:
            print("No frequent itemsets to visualize.")
            return None

        # Create visualization
        fig = px.bar(
            df,
            x='Itemset',
            y='Support Count',
            color='Size',
            hover_data=['Support %'],
            title='Frequent Itemsets by Support Count',
            labels={'Support Count': 'Number of Transactions'},
            height=600,
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title='Itemsets',
            yaxis_title='Support Count',
            xaxis_tickangle=-45
        )

        return fig

    def visualize_association_rules(self, top_n=20):
        """
        Visualize the association rules discovered by the algorithm.

        Parameters:
        top_n (int): Number of top rules to display

        Returns:
        plotly.graph_objects.Figure: Interactive visualization of association rules
        """
        if not self.rules:
            print("No rules to visualize. Run the algorithm first.")
            return None

        # Prepare data for visualization
        data = []

        # Sort rules by confidence
        sorted_rules = sorted(self.rules, key=lambda x: x[2], reverse=True)

        # Take top N rules
        for antecedent, consequent, confidence in sorted_rules[:top_n]:
            # Convert frozensets to string representation
            antecedent_str = ', '.join(map(str, antecedent))
            consequent_str = ', '.join(map(str, consequent))

            # Calculate support for the rule (support of antecedent ∪ consequent)
            rule_support = self._find_support(antecedent.union(consequent)) / self.transaction_count

            # Calculate lift
            consequent_support = self._find_support(consequent) / self.transaction_count
            lift = confidence / consequent_support if consequent_support > 0 else 0

            data.append({
                'Antecedent': antecedent_str,
                'Consequent': consequent_str,
                'Confidence': confidence,
                'Support': rule_support,
                'Lift': lift,
                'Rule': f"{antecedent_str} → {consequent_str}"
            })

        # Create DataFrame
        df = pd.DataFrame(data)

        if len(df) == 0:
            print("No rules to visualize.")
            return None

        # Create scatterplot
        fig = px.scatter(
            df,
            x='Support',
            y='Confidence',
            size='Lift',
            color='Lift',
            hover_data=['Antecedent', 'Consequent'],
            text='Rule',
            title='Association Rules by Support, Confidence and Lift',
            labels={
                'Support': 'Support (frequency of the rule)',
                'Confidence': 'Confidence (reliability of the rule)'
            },
            height=600,
            color_continuous_scale='Viridis'
        )

        fig.update_traces(
            textposition='top center',
            marker=dict(sizemin=5, sizeref=2)
        )

        fig.update_layout(
            xaxis_title='Support',
            yaxis_title='Confidence'
        )

        return fig

    def visualize_itemset_distribution(self):
        """
        Visualize the distribution of frequent itemsets by size.

        Returns:
        plotly.graph_objects.Figure: Bar chart showing itemset distribution
        """
        sizes = []
        counts = []

        for k, itemsets in self.frequent_itemsets.items():
            sizes.append(str(k))
            counts.append(len(itemsets))

        fig = go.Figure(data=[
            go.Bar(
                x=sizes,
                y=counts,
                marker_color='rgb(55, 83, 109)'
            )
        ])

        fig.update_layout(
            title='Distribution of Frequent Itemsets by Size',
            xaxis_title='Itemset Size',
            yaxis_title='Number of Frequent Itemsets',
            bargap=0.2
        )

        return fig

    def generate_candidates(self, prev_frequent_itemsets, k):
        """
        Generate candidate itemsets of size k from frequent itemsets of size k-1.

        Parameters:
        prev_frequent_itemsets (dict): Dictionary of frequent itemsets from previous iteration.
        k (int): The size of itemsets to generate.

        Returns:
        set: Generated candidate itemsets.
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
        candidates (set): Candidate itemsets to be pruned.
        prev_frequent_itemsets (dict): Previous frequent itemsets for reference.

        Returns:
        set: Pruned candidate itemsets.
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
        transactions (list): List of transaction sets.
        candidates (set): Set of candidate itemsets.

        Returns:
        dict: Dictionary mapping candidate itemsets to their absolute support counts.
        """
        support_counts = {itemset: 0 for itemset in candidates}

        for transaction in transactions:
            for itemset in candidates:
                if itemset.issubset(transaction):
                    support_counts[itemset] += 1

        # Return the raw counts (not divided by transaction count)
        return support_counts

    def eliminate_infrequent(self, candidates_support):
        """
        Eliminate candidates that do not meet the minimum support threshold.

        Parameters:
        candidates_support (dict): Dictionary mapping itemsets to their support counts.

        Returns:
        dict: Dictionary of itemsets that meet the support threshold.
        """
        # Create a new dictionary to store frequent itemsets
        frequent_itemsets = {}

        # Iterate through each candidate and its support count
        for itemset, count in candidates_support.items():
            # Check if the count meets the minimum threshold
            if count >= self.min_support:
                # If it does, add it to the frequent itemsets
                frequent_itemsets[itemset] = count

        return frequent_itemsets

    def find_frequent_itemsets(self, transactions):
        """
        Iteratively find frequent itemsets using the Apriori algorithm.

        Parameters:
        transactions (list): List of transaction sets.
        """
        start_time = time.time()
        self.transaction_count = len(transactions)

        # Find all unique items across all transactions
        unique_items = set()
        for transaction in transactions:
            unique_items.update(transaction)

        # Generate candidate 1-itemsets and count their support
        candidates_1 = {frozenset([item]): 0 for item in unique_items}

        # Time the support counting for 1-itemsets
        count_start = time.time()
        for transaction in transactions:
            for itemset in candidates_1:
                if itemset.issubset(transaction):
                    candidates_1[itemset] += 1
        count_time = time.time() - count_start
        self.execution_times['support_counting'] += count_time

        # Filter by minimum support to get frequent 1-itemsets
        self.frequent_itemsets[1] = self.eliminate_infrequent(candidates_1)

        # Record statistics for the first phase
        phase_stats = {
            'k': 1,
            'candidates_count': len(candidates_1),
            'frequent_count': len(self.frequent_itemsets[1]),
            'time': count_time
        }
        self.execution_times['phases'].append(phase_stats)

        k = 2
        while self.frequent_itemsets.get(k - 1):  # Continue until no more frequent itemsets found
            phase_start = time.time()

            # Generate candidate itemsets
            gen_start = time.time()
            candidates = self.generate_candidates(self.frequent_itemsets[k - 1], k)
            gen_time = time.time() - gen_start
            self.execution_times['candidate_generation'] += gen_time

            # Prune candidates using the Apriori property
            prune_start = time.time()
            candidates = self.prune_candidates(candidates, self.frequent_itemsets[k - 1])
            prune_time = time.time() - prune_start

            if not candidates:
                break

            # Count support for candidates
            count_start = time.time()
            candidates_support = self.count_support(transactions, candidates)
            count_time = time.time() - count_start
            self.execution_times['support_counting'] += count_time

            # Eliminate infrequent itemsets
            frequent_k = self.eliminate_infrequent(candidates_support)

            if frequent_k:
                self.frequent_itemsets[k] = frequent_k

            # Record statistics for this phase
            phase_time = time.time() - phase_start
            phase_stats = {
                'k': k,
                'candidates_count': len(candidates),
                'frequent_count': len(frequent_k) if frequent_k else 0,
                'time': phase_time,
                'candidate_generation_time': gen_time,
                'pruning_time': prune_time,
                'support_counting_time': count_time
            }
            self.execution_times['phases'].append(phase_stats)

            # Move to next level
            k += 1

        # Remove empty levels (though there shouldn't be any)
        self.frequent_itemsets = {k: v for k, v in self.frequent_itemsets.items() if v}

        # Record total time for finding frequent itemsets
        self.execution_times['frequent_itemsets'] = time.time() - start_time

    def generate_rules(self):
        """
        Generate association rules from the discovered frequent itemsets.

        Updates self.rules with generated rules as tuples (antecedent, consequent, confidence)
        """
        start_time = time.time()

        self.rules = []

        # Only consider itemsets of size 2 or larger
        for k in self.frequent_itemsets:
            if k < 2:
                continue

            for itemset, count in self.frequent_itemsets[k].items():
                # Generate all non-empty proper subsets as potential antecedents
                for i in range(1, k):
                    for antecedent in self._get_subsets(itemset, i):
                        consequent = itemset - antecedent

                        # Calculate confidence
                        antecedent_count = self._find_support(antecedent)
                        confidence = count / antecedent_count if antecedent_count > 0 else 0

                        if confidence >= self.min_confidence:
                            self.rules.append((antecedent, consequent, confidence))

        # Record total time for rule generation
        self.execution_times['rule_generation'] = time.time() - start_time

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
        Find the support count of an itemset from stored frequent itemsets.

        Parameters:
        itemset (frozenset): The itemset to find support for.

        Returns:
        int: Support count of the itemset.
        """
        k = len(itemset)
        return self.frequent_itemsets[k][itemset] if itemset in self.frequent_itemsets[k] else 0

    def run(self, transactions):
        """
        Execute the Apriori algorithm: find frequent itemsets and generate rules.

        Parameters:
        transactions (list): List of transactions.

        Returns:
        tuple: (frequent_itemsets, rules, execution_times)
        """
        # Start total timing
        total_start = time.time()

        # Convert transactions to sets if they aren't already
        transactions = [set(t) if not isinstance(t, set) else t for t in transactions]

        # Find frequent itemsets
        self.find_frequent_itemsets(transactions)

        # Generate rules
        self.generate_rules()

        # Record total execution time
        self.execution_times['total'] = time.time() - total_start

        return self.frequent_itemsets, self.rules, self.execution_times

    def visualize_execution_time(self):
        """
        Visualize the execution time for different phases of the algorithm.

        Returns:
        plotly.graph_objects.Figure: Bar chart of execution times
        """
        if not self.execution_times['total']:
            print("No timing information available. Run the algorithm first.")
            return None

        # Create figure for overall timing
        fig1 = go.Figure()

        # Add bars for the main phases
        phases = ['total', 'frequent_itemsets', 'candidate_generation', 'support_counting', 'rule_generation']
        times = [self.execution_times[phase] for phase in phases]

        fig1.add_trace(go.Bar(
            x=phases,
            y=times,
            marker_color='rgb(55, 83, 109)'
        ))

        fig1.update_layout(
            title='Execution Time Breakdown',
            xaxis_title='Phase',
            yaxis_title='Time (seconds)',
            bargap=0.2
        )

        # Create second figure for iteration-level timing
        if self.execution_times['phases']:
            phase_data = self.execution_times['phases']

            # Extract data for phases
            k_values = [phase['k'] for phase in phase_data]
            times = [phase['time'] for phase in phase_data]
            candidates = [phase['candidates_count'] for phase in phase_data]
            frequent = [phase['frequent_count'] for phase in phase_data]

            # Create dataframe
            df = pd.DataFrame({
                'Iteration': k_values,
                'Time (s)': times,
                'Candidates': candidates,
                'Frequent Itemsets': frequent
            })

            # Create figure
            fig2 = go.Figure()

            # Add bar for time
            fig2.add_trace(go.Bar(
                x=df['Iteration'],
                y=df['Time (s)'],
                name='Execution Time (s)',
                marker_color='rgb(55, 83, 109)'
            ))

            # Add lines for counts
            fig2.add_trace(go.Scatter(
                x=df['Iteration'],
                y=df['Candidates'],
                mode='lines+markers',
                name='Candidates Count',
                yaxis='y2',
                marker_color='rgb(219, 64, 82)',
                line=dict(width=3)
            ))

            fig2.add_trace(go.Scatter(
                x=df['Iteration'],
                y=df['Frequent Itemsets'],
                mode='lines+markers',
                name='Frequent Itemsets Count',
                yaxis='y2',
                marker_color='rgb(50, 171, 96)',
                line=dict(width=3)
            ))

            # Update layout for dual y-axis
            fig2.update_layout(
                title='Performance by Iteration',
                xaxis=dict(
                    title='Iteration (k)',
                    tickmode='array',
                    tickvals=df['Iteration']
                ),
                yaxis=dict(
                    title='Time (seconds)',
                    side='left'
                ),
                yaxis2=dict(
                    title='Count',
                    side='right',
                    overlaying='y',
                    rangemode='tozero'
                ),
                legend=dict(
                    x=0.1,
                    y=1.1,
                    orientation='h'
                ),
                bargap=0.2
            )

            return fig1, fig2

        return fig1


def print_summary_statistics(apriori, transactions, min_support, min_confidence, filename):
    """
    Print comprehensive summary statistics for the Apriori algorithm run.

    Parameters:
    apriori (Apriori): The Apriori object after running the algorithm
    transactions (list): List of transaction sets
    min_support (int): Minimum support threshold used
    min_confidence (float): Minimum confidence threshold used
    filename (str): Name of the input file
    """
    # Calculate basic dataset statistics
    unique_items = set()
    max_transaction_length = 0

    for transaction in transactions:
        unique_items.update(transaction)
        max_transaction_length = max(max_transaction_length, len(transaction))

    # Calculate minimum support percentage
    min_support_pct = (min_support / len(transactions)) * 100

    # Calculate frequent itemset statistics
    total_frequent_itemsets = sum(len(itemsets) for itemsets in apriori.frequent_itemsets.values())

    # Find highest confidence and lift rules
    highest_confidence_rule = None
    highest_lift_rule = None
    max_confidence = 0
    max_lift = 0

    for antecedent, consequent, confidence in apriori.rules:
        # Calculate lift for this rule
        antecedent_support = apriori._find_support(antecedent) / apriori.transaction_count
        consequent_support = apriori._find_support(consequent) / apriori.transaction_count
        rule_support = apriori._find_support(antecedent.union(consequent)) / apriori.transaction_count
        lift = rule_support / (
                    antecedent_support * consequent_support) if antecedent_support * consequent_support > 0 else 0

        if confidence > max_confidence:
            max_confidence = confidence
            highest_confidence_rule = (antecedent, consequent, confidence, lift)

        if lift > max_lift:
            max_lift = lift
            highest_lift_rule = (antecedent, consequent, confidence, lift)

    # Print summary statistics
    print("\n" + "=" * 50)
    print("APRIORI ALGORITHM SUMMARY")
    print("=" * 50)
    print(f"minsuppc: {min_support_pct:.2f}%")
    print(f"minconf: {min_confidence}")
    print(f"input file: {filename}")
    print(f"Number of items: {len(unique_items)}")
    print(f"Number of transactions: {len(transactions)}")
    print(f"The length of the longest transaction: {max_transaction_length}")

    print("\nFrequent Itemsets:")
    for k, itemsets in apriori.frequent_itemsets.items():
        print(f"Number of frequent {k}-itemsets: {len(itemsets)}")

    print(f"Total number of frequent itemsets: {total_frequent_itemsets}")
    print(f"Number of high-confidence rules: {len(apriori.rules)}")

    if highest_confidence_rule:
        antecedent, consequent, confidence, lift = highest_confidence_rule
        print(f"\nThe rule with the highest confidence ({confidence:.4f}):")
        print(f"  {{{', '.join(map(str, antecedent))}}} => {{{', '.join(map(str, consequent))}}}")

    if highest_lift_rule:
        antecedent, consequent, confidence, lift = highest_lift_rule
        print(f"\nThe rule with the highest lift ({lift:.4f}):")
        print(f"  {{{', '.join(map(str, antecedent))}}} => {{{', '.join(map(str, consequent))}}}")

    print(f"\nTime in seconds to find the frequent itemsets: {apriori.execution_times['frequent_itemsets']:.4f}")
    print(f"Time in seconds to find the confident rules: {apriori.execution_times['rule_generation']:.4f}")
    print("=" * 50)

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


def write_itemsets_to_file(apriori, filename="items.txt"):
    """
    Write frequent itemsets to a text file in the format:
    ITEMSETS|SUPPORT_COUNT|SUPPORT

    Parameters:
    apriori (Apriori): The Apriori object after running the algorithm
    filename (str): Name of the output file
    """
    with open(filename, 'w') as file:
        for k, itemsets in apriori.frequent_itemsets.items():
            for itemset, count in itemsets.items():
                # Convert itemset to space-delimited string
                items_str = ' '.join(map(str, itemset))

                # Calculate support as a fraction
                support = count / apriori.transaction_count

                # Write to file in the specified format
                file.write(f"{items_str}|{count}|{support:.6f}\n")

    print(f"Frequent itemsets written to {filename}")


def write_rules_to_file(apriori, filename="rules.txt"):
    """
    Write association rules to a text file in the format:
    LHS|RHS|SUPPORT_COUNT|SUPPORT|CONFIDENCE|LIFT

    Parameters:
    apriori (Apriori): The Apriori object after running the algorithm
    filename (str): Name of the output file
    """
    with open(filename, 'w') as file:
        for antecedent, consequent, confidence in apriori.rules:
            # Convert itemsets to space-delimited strings
            lhs_str = ' '.join(map(str, antecedent))
            rhs_str = ' '.join(map(str, consequent))

            # Calculate support counts and support
            rule_itemset = antecedent.union(consequent)
            support_count = apriori._find_support(rule_itemset)
            support = support_count / apriori.transaction_count

            # Calculate lift
            antecedent_support = apriori._find_support(antecedent) / apriori.transaction_count
            consequent_support = apriori._find_support(consequent) / apriori.transaction_count
            lift = confidence / consequent_support if consequent_support > 0 else 0

            # Write to file in the specified format
            file.write(f"{lhs_str}|{rhs_str}|{support_count}|{support:.6f}|{confidence:.6f}|{lift:.6f}\n")

    print(f"Association rules written to {filename}")


def write_info_to_file(apriori, transactions, min_support, min_confidence, input_filename, filename="info.txt"):
    """
    Write summary information to a text file.

    Parameters:
    apriori (Apriori): The Apriori object after running the algorithm
    transactions (list): List of transaction sets
    min_support (int): Minimum support threshold used
    min_confidence (float): Minimum confidence threshold used
    input_filename (str): Name of the input file
    filename (str): Name of the output file
    """
    # Calculate basic dataset statistics
    unique_items = set()
    max_transaction_length = 0

    for transaction in transactions:
        unique_items.update(transaction)
        max_transaction_length = max(max_transaction_length, len(transaction))

    # Calculate minimum support percentage
    min_support_pct = (min_support / len(transactions)) * 100

    # Find highest confidence and lift rules
    highest_confidence_rule = None
    highest_lift_rule = None
    max_confidence = 0
    max_lift = 0

    for antecedent, consequent, confidence in apriori.rules:
        # Calculate lift for this rule
        antecedent_support = apriori._find_support(antecedent) / apriori.transaction_count
        consequent_support = apriori._find_support(consequent) / apriori.transaction_count
        lift = confidence / consequent_support if consequent_support > 0 else 0

        if confidence > max_confidence:
            max_confidence = confidence
            highest_confidence_rule = (antecedent, consequent, confidence, lift)

        if lift > max_lift:
            max_lift = lift
            highest_lift_rule = (antecedent, consequent, confidence, lift)

    # Write to the info file
    with open(filename, 'w') as file:
        file.write(f"minsuppc: {min_support_pct:.2f}\n")
        file.write(f"minconf: {min_confidence}\n")
        file.write(f"input file: {input_filename}\n")
        file.write(f"Number of items: {len(unique_items)}\n")
        file.write(f"Number of transactions: {len(transactions)}\n")
        file.write(f"The length of the longest transaction: {max_transaction_length}\n")

        # Write frequent itemset counts by size
        total_frequent_itemsets = 0
        for k, itemsets in apriori.frequent_itemsets.items():
            count = len(itemsets)
            total_frequent_itemsets += count
            file.write(f"Number of frequent {k}-itemsets: {count}\n")

        file.write(f"Total number of frequent itemsets: {total_frequent_itemsets}\n")
        file.write(f"Number of high-confidence rules: {len(apriori.rules)}\n")

        # Write highest confidence rule
        if highest_confidence_rule:
            antecedent, consequent, confidence, lift = highest_confidence_rule
            file.write(
                f"The rule with the highest confidence: {{{', '.join(map(str, antecedent))}}} => {{{', '.join(map(str, consequent))}}}\n")

        # Write highest lift rule
        if highest_lift_rule:
            antecedent, consequent, confidence, lift = highest_lift_rule
            file.write(
                f"The rule with the highest lift: {{{', '.join(map(str, antecedent))}}} => {{{', '.join(map(str, consequent))}}}\n")

        # Write timing information
        file.write(
            f"Time in seconds to find the frequent itemsets: {apriori.execution_times['frequent_itemsets']:.4f}\n")
        file.write(f"Time in seconds to find the confident rules: {apriori.execution_times['rule_generation']:.4f}\n")

    print(f"Summary information written to {filename}")

if __name__ == "__main__":
    # Load transactions from file
    filename = 'small.txt'
    transactions = load_transactions_from_file(filename)

    # Print some transaction statistics
    print(f"Number of transactions: {len(transactions)}")
    transaction_count = len(transactions)

    # Define arrays of support counts and confidence levels to test
    # You can customize these arrays with your desired values
    support_values = [50, 75, 100, 125, 150]
    confidence_values = [0.6, 0.7, 0.8, 0.9]

    # Create a structure to store results for all combinations
    all_results = []

    print("\nRunning Apriori algorithm with multiple parameter combinations...")
    print(f"Support values: {support_values}")
    print(f"Confidence values: {confidence_values}")
    print(f"Total combinations to run: {len(support_values) * len(confidence_values)}")

    # Loop through all combinations of support and confidence
    for min_confidence in confidence_values:
        for min_support in support_values:
            # Calculate support percentage for reporting
            min_support_pct = (min_support / transaction_count) * 100

            print(
                f"\nRunning with min_support = {min_support} ({min_support_pct:.2f}%) and min_confidence = {min_confidence}")
            start_time = time.time()

            # Run the algorithm
            apriori = Apriori(min_support, min_confidence)
            frequent_itemsets, rules, execution_times = apriori.run(transactions)

            total_time = time.time() - start_time

            # Calculate total number of frequent itemsets
            total_frequent = sum(len(itemsets) for itemsets in frequent_itemsets.items())

            # Find highest confidence and lift rules if rules exist
            highest_confidence_rule = None
            highest_lift_rule = None
            max_confidence = 0
            max_lift = 0

            if rules:
                for antecedent, consequent, confidence in rules:
                    # Calculate lift for this rule
                    antecedent_support = apriori._find_support(antecedent) / transaction_count
                    consequent_support = apriori._find_support(consequent) / transaction_count
                    rule_support = apriori._find_support(antecedent.union(consequent)) / transaction_count
                    lift = rule_support / (
                                antecedent_support * consequent_support) if antecedent_support * consequent_support > 0 else 0

                    if confidence > max_confidence:
                        max_confidence = confidence
                        highest_confidence_rule = (antecedent, consequent, confidence, lift)

                    if lift > max_lift:
                        max_lift = lift
                        highest_lift_rule = (antecedent, consequent, confidence, lift)

            # Store results in a structured way
            result = {
                'min_support': min_support,
                'min_support_pct': min_support_pct,
                'min_confidence': min_confidence,
                'total_time': total_time,
                'frequent_itemsets_time': execution_times['frequent_itemsets'],
                'rule_generation_time': execution_times['rule_generation'],
                'frequent_itemsets_count': total_frequent,
                'rules_count': len(rules),
                'max_confidence': max_confidence if rules else 0,
                'max_lift': max_lift if rules else 0,
                'itemset_counts': {k: len(v) for k, v in frequent_itemsets.items()}
            }

            all_results.append(result)

            # Print a brief summary for this run
            print(f"  Found {total_frequent} frequent itemsets and {len(rules)} rules")
            print(f"  Total time: {total_time:.2f} seconds")

            # Run the full summary once for each combination if desired
            # Uncomment this line if you want the detailed summary for each run
            # print_summary_statistics(apriori, transactions, min_support, min_confidence, filename)

    # Print the comprehensive results table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY FOR ALL PARAMETER COMBINATIONS")
    print("=" * 80)
    print(
        f"{'Support':<8} {'Support %':<10} {'Conf':<6} {'Time(s)':<8} {'Itemsets':<8} {'Rules':<8} {'Max Conf':<8} {'Max Lift':<8}")
    print("-" * 80)

    # Sort by confidence level first, then by support count
    sorted_results = sorted(all_results, key=lambda x: (x['min_confidence'], x['min_support']))

    for result in sorted_results:
        print(f"{result['min_support']:<8} {result['min_support_pct']:.2f}%{' ' * 5} "
              f"{result['min_confidence']:.2f}{' ' * 2} "
              f"{result['total_time']:.2f}{' ' * 3} "
              f"{result['frequent_itemsets_count']:<8} "
              f"{result['rules_count']:<8} "
              f"{result['max_confidence']:.4f}{' ' * 2} "
              f"{result['max_lift']:.4f}")

    # Create visualizations for the multi-parameter results
    try:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Convert results to DataFrame for easier visualization
        results_df = pd.DataFrame(all_results)

        # Create a heatmap of frequent itemset counts
        fig1 = px.density_heatmap(
            results_df,
            x='min_support',
            y='min_confidence',
            z='frequent_itemsets_count',
            title='Number of Frequent Itemsets by Support and Confidence',
            labels={'min_support': 'Minimum Support Count',
                    'min_confidence': 'Minimum Confidence',
                    'frequent_itemsets_count': 'Number of Frequent Itemsets'}
        )

        # Create a heatmap of rule counts
        fig2 = px.density_heatmap(
            results_df,
            x='min_support',
            y='min_confidence',
            z='rules_count',
            title='Number of Association Rules by Support and Confidence',
            labels={'min_support': 'Minimum Support Count',
                    'min_confidence': 'Minimum Confidence',
                    'rules_count': 'Number of Rules'}
        )

        # Create a heatmap of execution times
        fig3 = px.density_heatmap(
            results_df,
            x='min_support',
            y='min_confidence',
            z='total_time',
            title='Execution Time by Support and Confidence',
            labels={'min_support': 'Minimum Support Count',
                    'min_confidence': 'Minimum Confidence',
                    'total_time': 'Execution Time (s)'}
        )

        # Create a 3D surface plot for a more visual representation
        fig4 = go.Figure(data=[
            go.Surface(
                z=results_df.pivot_table(index='min_confidence', columns='min_support',
                                         values='frequent_itemsets_count').values,
                x=sorted(results_df['min_support'].unique()),
                y=sorted(results_df['min_confidence'].unique()),
                colorscale='Viridis',
                colorbar=dict(title='Count')
            )
        ])

        fig4.update_layout(
            title='Frequent Itemsets (3D View)',
            scene=dict(
                xaxis_title='Support Count',
                yaxis_title='Confidence',
                zaxis_title='Number of Frequent Itemsets'
            )
        )

        # Show the figures
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()

        # Save the figures
        fig1.write_html("itemsets_by_params_heatmap.html")
        fig2.write_html("rules_by_params_heatmap.html")
        fig3.write_html("time_by_params_heatmap.html")
        fig4.write_html("itemsets_3d_surface.html")

        print("\nVisualization HTML files have been created.")

        # Optionally: Create a detailed line chart for each confidence level
        # This shows how itemset count and rule count change with support for each confidence level
        fig5 = make_subplots(rows=1, cols=2,
                             subplot_titles=("Frequent Itemsets by Support", "Association Rules by Support"),
                             shared_xaxes=True)

        for conf in sorted(results_df['min_confidence'].unique()):
            conf_data = results_df[results_df['min_confidence'] == conf]

            # Add itemsets trace
            fig5.add_trace(
                go.Scatter(
                    x=conf_data['min_support'],
                    y=conf_data['frequent_itemsets_count'],
                    mode='lines+markers',
                    name=f'Conf={conf} (Itemsets)',
                    line=dict(width=2)
                ),
                row=1, col=1
            )

            # Add rules trace
            fig5.add_trace(
                go.Scatter(
                    x=conf_data['min_support'],
                    y=conf_data['rules_count'],
                    mode='lines+markers',
                    name=f'Conf={conf} (Rules)',
                    line=dict(width=2, dash='dot')
                ),
                row=1, col=2
            )

        fig5.update_layout(
            title='Effect of Support and Confidence on Results',
            xaxis_title='Minimum Support Count',
            yaxis_title='Count'
        )

        fig5.show()
        fig5.write_html("support_confidence_effects.html")

    except Exception as e:
        print(f"\nError creating multi-parameter visualizations: {e}")
        print("Make sure you have Plotly and pandas installed:")
        print("pip install plotly pandas")

    # Run the detailed summary on the best parameter combination
    # (you can define "best" according to your criteria)
    print("\nShowing detailed summary for the parameter combination with the most rules:")
    best_result = max(all_results, key=lambda x: x['rules_count'])
    best_support = best_result['min_support']
    best_confidence = best_result['min_confidence']

    print(f"Best parameters: min_support={best_support}, min_confidence={best_confidence}")

    # Run one more time with the best parameters to generate detailed summary
    apriori = Apriori(best_support, best_confidence)
    frequent_itemsets, rules, execution_times = apriori.run(transactions)

    # Print detailed summary for the best parameter combination
    print_summary_statistics(apriori, transactions, best_support, best_confidence, filename)

    # Save results to files
    write_itemsets_to_file(apriori, "items.txt")
    write_rules_to_file(apriori, "rules.txt")
    write_info_to_file(apriori, transactions, min_support, min_confidence, filename, "info.txt")

    print("\nResults have been written to files: items.txt, rules.txt, and info.txt")