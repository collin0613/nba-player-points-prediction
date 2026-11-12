"""Durable credential-free ingestion artifacts and idempotent audit records."""
import hashlib
import json
from pathlib import Path
from django.conf import settings
from django.core.serializers.json import DjangoJSONEncoder
from core.models import DataImport


def record_import(source, kind, observed_at, scope, payload, status='success', diagnostics=None, row_count=0):
    raw = json.dumps(payload, cls=DjangoJSONEncoder, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    path = Path(settings.BASE_DIR) / 'data' / 'archive' / source / f'{digest}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(raw)
    item, created = DataImport.objects.get_or_create(source=source, kind=kind,
        observed_at=observed_at, request_fingerprint=fingerprint, defaults={
            'request_scope': scope, 'status': status, 'diagnostics': diagnostics or {}, 'row_count': row_count,
            'payload_path': str(path.relative_to(settings.BASE_DIR)), 'payload_sha256': digest})
    if not created and item.payload_sha256 != digest:
        raise ValueError('Conflicting payload for the same source observation; refusing to overwrite')
    return item, created
