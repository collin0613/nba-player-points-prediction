"""Run the reproducible historical rebuild, then fetch target-season rosters/schedule."""
import subprocess
import sys
from config import ROOT, PREDICT_SEASON, season_parser, season_value

def main():
    parser = season_parser(__doc__)
    parser.add_argument('--prediction-season', type=season_value, default=PREDICT_SEASON)
    parser.add_argument('--refresh', action='store_true', help='Replace downloaded season snapshots')
    parser.add_argument('--skip-context', action='store_true', help='Rebuild history/model only')
    args = parser.parse_args()
    for script in ['fetch_data', 'fetch_league_games', 'build_opponent_defense',
                   'process_data', 'build_dataset', 'train_model']:
        command = [sys.executable, str(ROOT / 'src' / f'{script}.py'), '--season', args.season]
        if args.refresh and script.startswith('fetch_'):
            command.append('--refresh')
        print(f'Running {script} ({args.season})', flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
    if not args.skip_context:
        command = [sys.executable, str(ROOT / 'src' / 'fetch_season_context.py'),
                   '--season', args.prediction_season]
        if args.refresh:
            command.append('--refresh')
        subprocess.run(command, check=True, cwd=ROOT)

if __name__ == '__main__':
    main()
