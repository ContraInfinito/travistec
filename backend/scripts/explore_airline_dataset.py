"""Quick explorer for the DelayedFlights.csv dataset.

Prints header, counts for delay/cancellation columns, unique origins/destinations,
and a small sample of rows. Designed to run even on large files by streaming.
"""
import csv
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parents[1] / \
    'datasets' / 'airline' / 'DelayedFlights.csv'


def main():
    if not P.exists():
        print(f"File not found: {P}")
        return

    with P.open('r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        print('COLUMNS:', len(header))
        print(header)

        # find indices for useful columns
        lower = [c.lower() for c in header]

        def idx(names):
            for n in names:
                if n in lower:
                    return lower.index(n)
            return None

        arr_idx = idx(['arrdelay', 'arr_delay',
                      'arr_delay_minutes', 'arr_delay_minutes'])
        dep_idx = idx(['depdelay', 'dep_delay', 'dep_delay_minutes'])
        cancelled_idx = idx(['cancelled', 'is_cancelled', 'cancelled_flag'])
        origin_idx = idx(['origin', 'originairport',
                         'originairportid', 'origincity'])
        dest_idx = idx(['dest', 'destination', 'destairport',
                       'destairportid', 'destinationcity'])

        counts = Counter()
        unique_origins = set()
        unique_dests = set()
        samples = []
        rows = 0
        for row in reader:
            rows += 1
            if arr_idx is not None:
                try:
                    v = float(row[arr_idx])
                    counts['arr_delay_count'] += 1
                    if v > 15:
                        counts['arr_delay_gt15'] += 1
                except Exception:
                    counts['arr_delay_na'] += 1
            if dep_idx is not None:
                try:
                    v = float(row[dep_idx])
                    counts['dep_delay_count'] += 1
                    if v > 15:
                        counts['dep_delay_gt15'] += 1
                except Exception:
                    counts['dep_delay_na'] += 1
            if cancelled_idx is not None:
                try:
                    v = row[cancelled_idx]
                    if v.strip() in ('1', 'True', 'true'):
                        counts['cancelled'] += 1
                except Exception:
                    pass
            if origin_idx is not None:
                unique_origins.add(row[origin_idx])
            if dest_idx is not None:
                unique_dests.add(row[dest_idx])
            if len(samples) < 5:
                samples.append(row)
            if rows % 500000 == 0:
                print(f'Processed {rows} rows...')

        print(f'Total rows processed: {rows}')
        print('Counts sample:', counts.most_common()[:10])
        print('Unique origins:', len(unique_origins))
        print('Unique dests:', len(unique_dests))
        print('Sample rows:')
        for s in samples:
            print(s[:20])

        print('\nSuggested prediction tasks:')
        print('- Regression: predict Arrival Delay (minutes) given DepDelay, distance, origin, dest, month, day')
        print('- Classification: predict Delay > 15 minutes (binary)')
        print('- Classification: predict Cancellation (cancelled field)')


if __name__ == '__main__':
    main()
