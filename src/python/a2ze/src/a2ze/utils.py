import hashlib
import io
import json
from typing import Any, Generator, Hashable, Mapping, Optional

import gffutils


def sha256sum(filename: str) -> str:
    # https://stackoverflow.com/a/44873382
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def create_unique_feature_id(feature: gffutils.Feature) -> str:
    if (feature.featuretype == 'gene') or (feature.featuretype == 'mRNA'):
        # don't mess up names of potential parents
        return feature.attributes['ID'][0]

    if 'ID' in feature.attributes:
        feature_id = feature.attributes['ID'][0]
    elif 'Name' in feature.attributes:
        feature_id = feature.attributes['Name'][0]
    else:
        feature_id = feature.attributes['Parent'][0] + "_" + feature.featuretype

    # features without children can have autoincremented IDs
    return f"autoincrement:{feature_id}"


def leaves(mapping: Mapping, parents: Optional[list[Hashable]] = None) -> Generator[tuple[tuple[str, ...], Any], None, None]:
    if parents is None:
        parents = []

    for key in mapping:
        value = mapping[key]
        if isinstance(value, Mapping):
            yield from leaves(value, parents = parents + [key])
        else:
            yield (tuple(parents + [key]), value)


def nested_to_flat(mapping: Mapping, separator: str = '/') -> dict[str, Any]:
    return {separator.join(path): value for path, value in leaves(mapping)}


def hash_dict_with_str_keys(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            obj = d,
            ensure_ascii = True,
            sort_keys = True
        ).encode('utf-8')
    ).hexdigest()
