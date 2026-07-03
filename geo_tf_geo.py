import math
import re
from dataclasses import dataclass


COORD_RE = re.compile(
    r"(?P<lat>[+-]?\d+\.\d+)\s*,\s*(?P<lon>[+-]?\d+\.\d+)"
)
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Coordinate:
    lat: float
    lon: float
    text: str


@dataclass(frozen=True)
class CoordinateMatch:
    coord: Coordinate
    span: tuple
    lat_span: tuple
    lon_span: tuple


def extract_first_coordinate(text):
    for match in COORD_RE.finditer(text):
        lat = float(match.group("lat"))
        lon = float(match.group("lon"))
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return CoordinateMatch(
                coord=Coordinate(lat=lat, lon=lon, text=match.group(0).strip()),
                span=match.span(),
                lat_span=match.span("lat"),
                lon_span=match.span("lon"),
            )
    raise ValueError(f"no valid coordinate found: {text!r}")


def haversine_km(coord, reference):
    lat = math.radians(float(coord.lat))
    lon = math.radians(float(coord.lon))
    lat0 = math.radians(float(reference.lat))
    lon0 = math.radians(float(reference.lon))
    d_lat = lat - lat0
    d_lon = lon - lon0
    hav = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(lat0) * math.cos(lat) * math.sin(d_lon / 2.0) ** 2
    )
    hav = min(max(hav, 0.0), 1.0)
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(hav))


def wgs84_degree_scales(lat):
    phi = math.radians(float(lat))
    sin_phi = math.sin(phi)
    denom = 1.0 - WGS84_E2 * sin_phi * sin_phi
    m = WGS84_A * (1.0 - WGS84_E2) / (denom ** 1.5)
    n = WGS84_A / math.sqrt(denom)
    d_lat = math.pi / 180.0 * m
    d_lon = math.pi / 180.0 * n * math.cos(phi)
    return d_lat, d_lon


def _number_scale_entries(text, span, axis, degree_scale):
    number = text[span[0] : span[1]]
    sign_len = 1 if number and number[0] in "+-" else 0
    body_start = span[0] + sign_len
    body = number[sign_len:]
    integer, fraction = body.split(".", 1)
    entries = []
    integer_scales = []

    for i, _ in enumerate(integer):
        char_index = body_start + i
        k = len(integer) - i - 1
        scale = degree_scale * (10.0 ** k)
        entries.append((char_index, scale, axis, "digit", k))
        integer_scales.append(scale)

    dot_index = body_start + len(integer)
    entries.append((dot_index, degree_scale, axis, "dot", 0))

    frac_start = dot_index + 1
    for i, _ in enumerate(fraction):
        char_index = frac_start + i
        k = -(i + 1)
        scale = degree_scale * (10.0 ** k)
        entries.append((char_index, scale, axis, "digit", k))

    return entries, integer_scales


def build_wgs84_char_weights(text, coord_match):
    d_lat, d_lon = wgs84_degree_scales(coord_match.coord.lat)
    entries = []
    integer_scales = []

    lat_entries, lat_integer = _number_scale_entries(
        text, coord_match.lat_span, "lat", d_lat
    )
    lon_entries, lon_integer = _number_scale_entries(
        text, coord_match.lon_span, "lon", d_lon
    )
    entries.extend(lat_entries)
    entries.extend(lon_entries)
    integer_scales.extend(lat_integer)
    integer_scales.extend(lon_integer)
    sign_cap = max(integer_scales)

    lat_text = text[coord_match.lat_span[0] : coord_match.lat_span[1]]
    lon_text = text[coord_match.lon_span[0] : coord_match.lon_span[1]]
    if lat_text and lat_text[0] in "+-":
        scale = min(2.0 * abs(coord_match.coord.lat) * d_lat, sign_cap)
        entries.append((coord_match.lat_span[0], scale, "lat", "sign", None))
    if lon_text and lon_text[0] in "+-":
        delta_lon = min(
            2.0 * abs(coord_match.coord.lon),
            360.0 - 2.0 * abs(coord_match.coord.lon),
        )
        scale = min(delta_lon * d_lon, sign_cap)
        entries.append((coord_match.lon_span[0], scale, "lon", "sign", None))

    raw = [0.0 for _ in text]
    meta = []
    for char_index, scale, axis, kind, place in entries:
        raw[char_index] = math.log1p(float(scale))
        meta.append(
            {
                "char_index": char_index,
                "char": text[char_index],
                "axis": axis,
                "kind": kind,
                "place": place,
                "scale_m": float(scale),
                "log_weight": raw[char_index],
            }
        )

    active = [item for item in raw if item > 0.0]
    mean_weight = sum(active) / len(active)
    weights = [0.0 if item == 0.0 else item / mean_weight for item in raw]
    return weights, meta


def build_token_weights(tokenizer, answer_ids, answer_text):
    import torch

    coord_match = extract_first_coordinate(answer_text)
    char_weights, char_meta = build_wgs84_char_weights(answer_text, coord_match)
    ids = answer_ids[0] if answer_ids.dim() == 2 else answer_ids

    token_weights = []
    token_meta = []
    cursor = 0
    for token_index, token_id in enumerate(ids.tolist()):
        piece = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if piece == "":
            token_weights.append(0.0)
            token_meta.append(
                {
                    "token_index": token_index,
                    "token_id": int(token_id),
                    "decoded": piece,
                    "span": [cursor, cursor],
                    "weight": 0.0,
                }
            )
            continue
        start = answer_text.find(piece, cursor)
        if start < 0:
            raise ValueError(
                f"token piece {piece!r} cannot be aligned in {answer_text!r}"
            )
        end = start + len(piece)
        piece_weights = char_weights[start:end]
        weight = sum(piece_weights) / len(piece_weights)
        token_weights.append(float(weight))
        token_meta.append(
            {
                "token_index": token_index,
                "token_id": int(token_id),
                "decoded": piece,
                "span": [start, end],
                "weight": float(weight),
            }
        )
        cursor = end

    return {
        "coordinate": coord_match.coord,
        "coordinate_span": [coord_match.span[0], coord_match.span[1]],
        "lat_span": [coord_match.lat_span[0], coord_match.lat_span[1]],
        "lon_span": [coord_match.lon_span[0], coord_match.lon_span[1]],
        "char_weights": char_weights,
        "char_meta": char_meta,
        "token_weights": torch.tensor(token_weights, dtype=torch.float32).unsqueeze(0),
        "token_meta": token_meta,
    }
