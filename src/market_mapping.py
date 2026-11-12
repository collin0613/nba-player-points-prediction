"""Conservative identity normalization shared by sportsbook import and NBA results."""
import re
import unicodedata
from datetime import datetime
from nba_api.stats.static import teams


def normalize_player_name(value):
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().casefold()
    words = re.sub(r"[.\-'’]", '', value).split()
    while words and words[-1] in {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}:
        words.pop()
    return re.sub(r'[^a-z0-9]', '', ''.join(words))


TEAM_NAMES = {}
for team in teams.get_teams():
    for name in (team['full_name'], team['abbreviation']):
        TEAM_NAMES[normalize_player_name(name)] = team['abbreviation']
TEAM_NAMES.update({normalize_player_name(k): v for k, v in {
    'LA Clippers': 'LAC', 'GS': 'GSW', 'NO': 'NOP', 'NY': 'NYK', 'SA': 'SAS',
    'PHO': 'PHX', 'BRK': 'BKN', 'CHO': 'CHA'}.items()})


def normalize_team(value):
    return TEAM_NAMES.get(normalize_player_name(value))


def provider_team(info):
    names = info.get('names') or {}
    return normalize_team(names.get('short', '')) or normalize_team(names.get('long', ''))


def aware_datetime(value):
    if not isinstance(value, str):
        raise ValueError('Expected an ISO timestamp')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timestamp must include a timezone')
    return dt
