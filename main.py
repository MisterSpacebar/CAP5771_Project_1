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


# Usage with the provided dataset
if __name__ == "__main__":
    # Load transactions from file
    transactions = load_transactions_from_file('small.txt')

    # Print some transaction statistics
    print(f"Number of transactions: {len(transactions)}")
    print(f"Sample transactions (first 3):")
    for i, transaction in enumerate(transactions[:3]):
        print(f"  Transaction {i}: {transaction}")

    # Run Apriori with absolute support count
    min_support = 80  # Items must appear in at least 80 transactions
    min_confidence = 0.8  # 70% confidence in rules

    print("\nRunning Apriori algorithm...")
    start_time = time.time()

    apriori = Apriori(min_support, min_confidence)
    frequent_itemsets, rules, execution_times = apriori.run(transactions)

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

    # Print timing results
    print("\nExecution Time Breakdown:")
    print(f"  Finding frequent itemsets: {execution_times['frequent_itemsets']:.2f} seconds")
    print(f"  Candidate generation: {execution_times['candidate_generation']:.2f} seconds")
    print(f"  Support counting: {execution_times['support_counting']:.2f} seconds")
    print(f"  Rule generation: {execution_times['rule_generation']:.2f} seconds")

    # Print results
    print("\nFrequent Itemsets:")
    for k, itemsets in frequent_itemsets.items():
        print(f"  {k}-itemsets: {len(itemsets)}")
        # Print a few examples
        examples = list(itemsets.items())[:3]
        for itemset, count in examples:
            # Calculate support percentage for reference
            support_percentage = count / len(transactions)
            print(f"    {itemset} (count: {count}, support: {support_percentage:.3f})")

    print("\nAssociation Rules (top 5):")
    # Sort rules by confidence
    rules.sort(key=lambda x: x[2], reverse=True)
    for antecedent, consequent, confidence in rules[:5]:
        print(f"  {antecedent} => {consequent} (confidence: {confidence:.3f})")

    # Visualize the results
    try:
        # Create visualizations
        itemset_dist_fig = apriori.visualize_itemset_distribution()
        frequent_itemsets_fig = apriori.visualize_frequent_itemsets(top_n=15)
        rules_fig = apriori.visualize_association_rules(top_n=15)

        # Create execution time visualizations
        time_figs = apriori.visualize_execution_time()
        if isinstance(time_figs, tuple):
            time_overview_fig, time_detail_fig = time_figs
        else:
            time_overview_fig = time_figs
            time_detail_fig = None

        # Show the figures
        itemset_dist_fig.show()
        frequent_itemsets_fig.show()
        rules_fig.show()
        time_overview_fig.show()
        if time_detail_fig:
            time_detail_fig.show()

        # Save the figures
        itemset_dist_fig.write_html("itemset_distribution.html")
        frequent_itemsets_fig.write_html("frequent_itemsets.html")
        rules_fig.write_html("association_rules.html")
        time_overview_fig.write_html("execution_time_overview.html")
        if time_detail_fig:
            time_detail_fig.write_html("execution_time_by_iteration.html")

        print("\nVisualization HTML files have been created.")
    except Exception as e:
        print(f"\nError creating visualizations: {e}")
        print("Make sure you have Plotly and pandas installed:")
        print("pip install plotly pandas")

    # Benchmark with different support thresholds
    print("\nBenchmarking with different support thresholds...")
    support_values = [50, 75, 100, 125, 150, 200]
    benchmark_results = []

    for min_support in support_values:
        print(f"  Running with min_support = {min_support}...")
        start_time = time.time()

        apriori = Apriori(min_support, min_confidence)
        frequent_itemsets, rules, execution_times = apriori.run(transactions)

        total_time = time.time() - start_time

        # Record results
        benchmark_results.append({
            'min_support': min_support,
            'total_time': total_time,
            'frequent_itemsets_count': sum(len(itemsets) for itemsets in frequent_itemsets.values()),
            'rules_count': len(rules)
        })

    # Print benchmark results
    print("\nBenchmark Results:")
    print(f"{'Support':<10} {'Time (s)':<10} {'Itemsets':<10} {'Rules':<10}")
    print("-" * 40)
    for result in benchmark_results:
        print(f"{result['min_support']:<10} {result['total_time']:.2f}s{' ' * 4} "
              f"{result['frequent_itemsets_count']:<10} {result['rules_count']:<10}")

    # Visualize benchmark results
    try:
        # Create dataframe
        df = pd.DataFrame(benchmark_results)

        # Create figure
        fig = go.Figure()

        # Add bar for time
        fig.add_trace(go.Bar(
            x=df['min_support'],
            y=df['total_time'],
            name='Execution Time (s)',
            marker_color='rgb(55, 83, 109)'
        ))

        # Add lines for counts
        fig.add_trace(go.Scatter(
            x=df['min_support'],
            y=df['frequent_itemsets_count'],
            mode='lines+markers',
            name='Frequent Itemsets',
            yaxis='y2',
            marker_color='rgb(219, 64, 82)',
            line=dict(width=3)
        ))

        fig.add_trace(go.Scatter(
            x=df['min_support'],
            y=df['rules_count'],
            mode='lines+markers',
            name='Association Rules',
            yaxis='y2',
            marker_color='rgb(50, 171, 96)',
            line=dict(width=3)
        ))

        # Update layout for dual y-axis
        fig.update_layout(
            title='Performance vs. Support Threshold',
            xaxis=dict(
                title='Minimum Support Threshold',
                tickmode='array',
                tickvals=df['min_support']
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
            )
        )

        fig.show()
        fig.write_html("benchmark_results.html")
        print("Benchmark visualization has been created.")
    except Exception as e:
        print(f"\nError creating benchmark visualization: {e}")