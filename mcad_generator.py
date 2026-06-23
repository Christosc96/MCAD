import numpy as np
from itertools import cycle, combinations
import random
import stumpy
import math
from stumpy import config
import tqdm
import pandas as pd






config.STUMPY_EXCL_ZONE_DENOM = 1

class TimeSeriesGenerator:
    """Handles time series data generation"""

    def __init__(self):
        self.template_functions = [
            self.simple_on,
            self.step_recovery_noisy,
            self.on_off_cycle_ringing,
            self.prolonged_on_noisy,
            self.scada_spike,
        ]
        self.template_names = ["On Function", "Step Recovery", "On-Off Ringing", "Prolonged On", "Temperature Spike"]

    def add_template(self, name, template_func):
        """Add a custom template function

        Args:
            name: Name of the template
            template_func: Function that takes (length, **kwargs) and returns np.array
        """
        self.template_functions.append(template_func)
        self.template_names.append(name)
        return len(self.template_functions) - 1

    def simple_on(self, length, amplitude=1.0):
        noise = np.random.normal(0, 0.01, length)
        t = np.full(length, amplitude)
        return t + noise

    def scada_spike(self, length, amplitude=1.0):
        t = np.linspace(0, 1, length)
        rise = 1 / (1 + np.exp(-80 * (t - 0.25)))
        decay = np.exp(-25 * (t - 0.3)) * (t > 0.3)
        spike = rise * decay
        ringing = 0.05 * np.sin(40 * np.pi * t) * np.exp(-20 * (t - 0.3))
        ringing *= (t > 0.3)
        return amplitude * (spike + ringing)

    def step_recovery_noisy(self, length, amplitude=1.0, noise_std=0.2, seed=None):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, length)
        step = (t > 0.2).astype(float)
        recovery = np.exp(-6 * (t - 0.2)) * (t > 0.2)
        clean = amplitude * step * recovery
        noise = rng.normal(0, noise_std, length)
        return clean + noise

    def on_off_cycle_ringing(self, length, amplitude=2):
        t = np.linspace(0, 1, length)
        rise = 1 / (1 + np.exp(-12 * (t - 0.2)))
        fall = 1 / (1 + np.exp(-12 * (t - 0.6)))
        base = amplitude * (rise - fall)
        ringing = 0.12 * np.sin(18 * np.pi * t) * np.exp(-10 * (t - 0.2))
        ringing *= (t > 0.2)
        return base + ringing

    def prolonged_on_noisy(self, length, amplitude=1.0, noise_std=0.2, seed=None):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, length)
        rise = 1 / (1 + np.exp(-10 * (t - 0.15)))
        noise = np.random.normal(0, 0.01, length)
        return amplitude * rise + noise

    def generate_data(self, T=5000, N=4, k=3, random_templates=True, discord_length=50, normality_coef=2, min_gaps=[]):
        """Generate time series data with events"""
        np.random.seed(np.random.randint(0, 10000))

        event_lengths = [discord_length] * N
        min_gaps = [50] * N

        if random_templates:
            rng_template = np.random.default_rng()
            channel_template_indices = rng_template.choice(len(self.template_functions), size=N, replace=True)
        else:
            channel_template_indices = [0] * N

        templates = []
        for i in range(N):
            template_func = self.template_functions[channel_template_indices[i]]
            templates.append(template_func(event_lengths[i], amplitude=2.0))

        min_gaps = [50] * N

        series = np.random.normal(0, 0.01, (N, T))
        events_log = []

        subsets = []
        for k_c in range(2, k):
            subsets = subsets + list(combinations(range(N), k_c))

        random.shuffle(subsets)

        rng = np.random.default_rng()
        k_series = rng.choice(N, size=k, replace=False)

        forced_t = T // 2
        k_way_start = forced_t
        k_way_end = forced_t + max(event_lengths[i] for i in k_series)

        for t, subset in zip(
            range(0, T - event_lengths[0] + min_gaps[0], (min_gaps[0] + event_lengths[0])),
            cycle(subsets)
        ):
            if k_way_start <= t < k_way_end:
                continue

            max_end = t
            for channel_idx in subset:
                L = event_lengths[channel_idx]
                end = min(T, t + L)
                series[channel_idx, t:end] += templates[channel_idx][:end - t]
                max_end = max(max_end, end)

            events_log.append((t, max_end, frozenset(subset)))

        max_end = forced_t
        for i in k_series:
            L = event_lengths[i]
            end = forced_t + L
            series[i, forced_t:end] += templates[i]
            max_end = max(max_end, end)

        events_log.append((forced_t, max_end, frozenset(k_series)))

        warning_message = None
        available_space = T - discord_length + min_gaps[0]
        if available_space < normality_coef * len(subsets) * (discord_length + min_gaps[0]):
            warning_message = (
                f"Warning: Series too short for optimal event distribution! "
                f"Available space ({available_space}) is less than "
                f"required space for normal events pattern "
                f"({normality_coef * len(subsets) * (discord_length + min_gaps[0])}). "
                f"Consider increasing Series Length or decreasing Channels/Arity."
            )

        return {
            'series': series,
            'events_log': events_log,
            'k_way_start': k_way_start,
            'k_way_end': k_way_end,
            'k_series': list(k_series),
            'channel_template_indices': list(channel_template_indices),
            'template_names': [self.template_names[i] for i in channel_template_indices],
            'warning': warning_message,
        }


# --- Discord Profile Analysis ---

def flatten_sliding_windows(X, W, stride=1):
    """Flatten sliding windows for multivariate time series"""
    d, N = X.shape
    windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=W, axis=1)
    windows = windows[:, ::stride, :]  # (d, num_windows, W)
    return windows.transpose(1, 0, 2).reshape(windows.shape[1], d * W).flatten()


def subsets(S):
    """Generate all proper subsets of S (excluding S itself)"""
    result = []
    for r in range(len(S)):
        for c in combinations(S, r):
            result.append(set(c))
    return result


def supersets(S, n):
    """Generate all supersets of S within universe of size n"""
    S = set(S)
    U = set(range(n))
    remaining = list(U - S)
    result = []
    for r in range(len(remaining) + 1):
        for c in combinations(remaining, r):
            result.append(S | set(c))
    return result


def discord_profile(S, n_channels):
    """Generate discord profile: all subsets and supersets of S"""
    prof = subsets(S) + supersets(S, n_channels)
    print(prof)
    return [list(s) for s in prof]


def get_discord_score(X, subset=None, m=50):
    """Calculate discord score for a channel subset"""
    X_sub = X[subset, :] if subset is not None and len(subset) > 0 else X
    new_m = m * X_sub.shape[0]
    flattened = flatten_sliding_windows(X_sub, m, m)

    if flattened.shape[0] < 2:
        return 0.0

    flattened = flattened.astype(np.float32)

    matrix_profile = stumpy.stump(flattened, m=new_m, normalize=False)
    top_k_idx = np.argsort(matrix_profile[:, 0] * math.sqrt(1 / new_m))[-1]
    return matrix_profile[top_k_idx, 0] * math.sqrt(1 / new_m)


def run_discord_analysis(series, selected_channels, n_channels, m=50):
    """Run discord profile analysis on selected channels.

    Args:
        series:            np.ndarray of shape (N, T)
        selected_channels: list of channel indices to analyse
        n_channels:        total number of channels (N)
        m:                 window size

    Returns:
        dict with keys 'all_results', 'mean_scores', 'std_scores'
    """
    prof = discord_profile(selected_channels, n_channels)
    print(prof)
    scores_by_arity = {}
    all_results = []

    for subset in tqdm.tqdm(prof):
        if len(subset) == 0:
            continue
        try:
            score = get_discord_score(series, subset=subset, m=m)
            arity = len(subset)
            scores_by_arity.setdefault(arity, []).append(score)
            all_results.append({'subset': subset, 'arity': arity, 'score': score})
        except Exception as e:
            print(f"Error calculating score for subset {subset}: {e}")

    mean_scores = {a: np.mean(v) for a, v in scores_by_arity.items()}
    std_scores  = {a: np.std(v)  for a, v in scores_by_arity.items()}

    return {
        'all_results': all_results,
        'mean_scores': mean_scores,
        'std_scores':  std_scores,
    }

def save_analysis_as_dataframe(analysis, n_channels,
                               output_file="discord_dataset.parquet"):
    """
    Convert analysis['all_results'] into a binary-channel dataframe.

    Columns:
        ch0, ch1, ..., ch(N-1), arity, score
    """

    rows = []

    for result in analysis['all_results']:
        subset = set(result['subset'])

        row = {
            f'ch{i}': int(i in subset)
            for i in range(n_channels)
        }

        row['arity'] = result['arity']
        row['score'] = result['score']

        rows.append(row)

    df = pd.DataFrame(rows)

    if output_file.endswith(".csv"):
        df.to_csv(output_file, index=False)
    else:
        df.to_parquet(output_file, index=False)

    print(f"Saved dataset with {len(df)} rows to {output_file}")

    return df

# --- Main ---

if __name__ == '__main__':
    # Parameters
    T              = 10000
    N              = 4
    k              = 3
    discord_length = 50
    normality_coef = 2
    m              = 50            # window size for analysis
    selected_channels = [0, 1, 2]  # channels to analyse

    # hacky way to generate a full dataset that can be then downloaded

    selected_channels = list(range(N))


    # Generate data
    gen    = TimeSeriesGenerator()
    result = gen.generate_data(T=T, N=N, k=k, random_templates=True,
                               discord_length=discord_length,
                               normality_coef=normality_coef)

    if result['warning']:
        print(result['warning'])

    series = result['series']
    print(f"\nGenerated {N}-channel series of length {T}")
    print(f"Templates: {result['template_names']}")
    print(f"k-way anomaly channels : {result['k_series']}")
    print(f"k-way anomaly window   : [{result['k_way_start']}, {result['k_way_end']})")

    # Discord analysis
    print(f"\nRunning discord profile analysis on channels {selected_channels} with m={m}...")
    analysis = run_discord_analysis(series, selected_channels, N, m=m)

    print("\nMean anomaly scores by arity:")
    for arity in sorted(analysis['mean_scores']):
        mean = analysis['mean_scores'][arity]
        std  = analysis['std_scores'][arity]
        print(f"  Arity {arity}: mean={mean:.4f}  std={std:.4f}")

    print("\nTop 5 subsets by score:")
    top5 = sorted(analysis['all_results'], key=lambda x: x['score'], reverse=True)[:5]
    for r in top5:
        print(f"  {sorted(r['subset'])}  arity={r['arity']}  score={r['score']:.4f}")


    #save results in dataset

    df = save_analysis_as_dataframe(
    analysis,
    n_channels=N,
    output_file="discord_dataset.csv"
)