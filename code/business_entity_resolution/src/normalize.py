"""Text normalisation for business names and addresses.

Everything here is country-agnostic: rules are keyed on tokens, never on a
closed set of country labels, so unseen countries (e.g. France in test) flow
through the same code path.
"""
import re
import unicodedata

from anyascii import anyascii

# Indic scripts (Devanagari .. Malayalam) — handled with a learned word dictionary
# first and anyascii as a fallback.
SCRIPT_RE = re.compile(r"[ऀ-෿]")
SCRIPT_TOKEN_RE = re.compile(r"[ऀ-෿‌‍]+")

# ---------------------------------------------------------------- legal forms
# canonical legal-form token -> variants (already lower-cased, punctuation-free)
_LEGAL = {
    "pvt": ["pvt", "private", "pvtltd"],
    "ltd": ["ltd", "limited", "ltda", "lt"],
    "inc": ["inc", "incorporated", "incorporation"],
    "corp": ["corp", "corporation", "corpn"],
    "co": ["co", "company", "cos", "compnay"],
    "llc": ["llc", "l l c"],
    "llp": ["llp", "l l p"],
    "lp": ["lp"],
    "plc": ["plc"],
    "pllc": ["pllc"],
    "pc": ["pc"],
    "pa": ["pa"],
    "public": ["public"],
    "group": ["group", "grp"],
    "holdings": ["holdings", "holding"],
    "enterprises": ["enterprises", "enterprise"],
    # French / generic European forms
    "sarl": ["sarl"],
    "sas": ["sas"],
    "sasu": ["sasu"],
    "eurl": ["eurl"],
    "sa": ["sa"],
    "snc": ["snc"],
    "sci": ["sci"],
    "ei": ["ei"],
    "gmbh": ["gmbh"],
    "bv": ["bv"],
    "ag": ["ag"],
}
LEGAL_MAP = {}
for canon, vs in _LEGAL.items():
    for v in vs:
        LEGAL_MAP[v] = canon
# tokens that are pure legal form (removed from the "core" name)
LEGAL_CORE_DROP = {
    "pvt", "ltd", "inc", "corp", "co", "llc", "llp", "lp", "plc", "pllc", "pc",
    "pa", "public", "sarl", "sas", "sasu", "eurl", "sa", "snc", "ei", "gmbh",
    "bv", "ag", "sci", "the", "and", "of", "dba", "france", "india", "usa", "us",
}

NON_LEGAL = {"the", "and", "of", "dba", "france", "india", "usa", "us"}

# ------------------------------------------------------------ address tokens
_ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "avn": "avenue", "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard",
    "dr": "drive", "drv": "drive", "ct": "court", "crt": "court", "ln": "lane",
    "pl": "place", "hwy": "highway", "pkwy": "parkway", "pky": "parkway",
    "cir": "circle", "trl": "trail", "ter": "terrace", "terr": "terrace",
    "sq": "square", "mt": "mount", "ft": "fort", "jct": "junction",
    "expy": "expressway", "fwy": "freeway", "tpke": "turnpike", "aly": "alley",
    "cv": "cove", "pt": "point", "xing": "crossing", "rte": "route", "rt": "route",
    "apt": "apartment", "apts": "apartments", "ste": "suite", "fl": "floor",
    "flr": "floor", "bldg": "building", "blg": "building", "rm": "room",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast",
    "nw": "northwest", "se": "southeast", "sw": "southwest",
    "nr": "near", "opp": "opposite", "bh": "behind", "nagr": "nagar",
    "marg": "marg", "clny": "colony", "col": "colony", "sec": "sector",
    "sect": "sector", "ph": "phase", "extn": "extension", "ext": "extension",
    "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluka",
    "r": "rue", "imp": "impasse", "all": "allee", "chem": "chemin",
    "pte": "porte", "fbg": "faubourg", "rte": "route", "gal": "galerie",
    "mal": "marechal", "gen": "general", "gal.": "general", "st.": "saint",
    "ste.": "sainte", "pdt": "president",
}
_ORD = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "eleventh": "11th", "twelfth": "12th",
}
ADDR_MAP = dict(_ADDR_ABBR)
ADDR_MAP.update(_ORD)

# house-number prefixes that carry no information
ADDR_FILLER = {
    "no", "nos", "number", "num", "h", "hno", "house", "door", "dno", "plot",
    "unit", "shop", "flat", "fno", "sno", "sr", "survey", "khasra", "kh", "n",
    "null", "na", "none", "nan", "bis", "ter", "and", "the", "of", "de", "du",
    "des", "la", "le", "les", "d", "l", "po", "box", "c", "o", "at", "post",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka",
    "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "ladakh": "la",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
    "andaman and nicobar islands": "an",
}
# component (full string) -> canonical state tag. Abbreviations are only
# interpreted when they stand alone as an address component.
STATE_COMPONENT = {}
for full, ab in US_STATES.items():
    STATE_COMPONENT[full] = "st_" + ab
for full, ab in IN_STATES.items():
    STATE_COMPONENT[full] = "st_in_" + ab
STATE_ABBR_COMPONENT = {}
for full, ab in US_STATES.items():
    STATE_ABBR_COMPONENT.setdefault(ab, "st_" + ab)
IN_ABBR = {v: "st_in_" + v for v in IN_STATES.values()}
IN_ABBR.update({"ts": "st_in_tg", "or": "st_in_od", "uk": "st_in_uk", "ut": "st_in_uk"})

CITY_ALIAS = {
    "bombay": "mumbai", "bangalore": "bengaluru", "gurgaon": "gurugram",
    "madras": "chennai", "calcutta": "kolkata", "poona": "pune", "baroda": "vadodara",
    "trivandrum": "thiruvananthapuram", "mysore": "mysuru", "cochin": "kochi",
    "benares": "varanasi", "allahabad": "prayagraj", "new delhi": "delhi",
}

# leetspeak / OCR style digit-for-letter substitutions inside alphabetic words
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
_WORD_RE = re.compile(r"[a-z0-9]+")
_MIXED_RE = re.compile(r"^(?=.*[a-z])(?=.*[0-9])[a-z0-9]+$")
_DBA_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|d\.b\.a\.?|doing business as|t/a|trading as|aka|a/k/a)\b")
_WEB_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|co|fr|biz|info|io|us)\b")
_DOTTED_RE = re.compile(r"(?<![a-z0-9])(?:[a-z]\.){2,}[a-z]?\.?(?![a-z0-9])")
_GLUED_RE = re.compile(r"\b(no|nos|hno|dno|fno|sno|plot|flat|door|house|shop|unit|apt|ste|suite|bldg)(\d)")
_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")


class Transliterator:
    """Word-level dictionary for Indic-script tokens learned from training pairs,
    with anyascii as fallback for unseen words."""

    def __init__(self, word_map=None, comp_map=None):
        self.word_map = word_map or {}
        self.comp_map = comp_map or {}

    def __call__(self, text, is_addr=False):
        if not SCRIPT_RE.search(text):
            return text
        if is_addr and self.comp_map:
            parts = [p.strip() for p in text.split(",")]
            parts = [self.comp_map.get(p, p) if SCRIPT_RE.search(p) else p for p in parts]
            text = ", ".join(parts)
            if not SCRIPT_RE.search(text):
                return text
        wm = self.word_map
        return SCRIPT_TOKEN_RE.sub(lambda m: " " + wm.get(m.group(0), anyascii(m.group(0))) + " ", text)


def _basic(text):
    text = unicodedata.normalize("NFKC", text).replace("\u00b0", " ").replace("\u00ba", " ")
    text = anyascii(text).lower()
    return text


def _ordinal(n):
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _fix_token(t):
    if _MIXED_RE.match(t) and not t[0].isdigit():
        # letters with embedded digits -> probable OCR/leet noise ("chemica1s")
        return t.translate(_LEET)
    if _MIXED_RE.match(t) and t[0].isdigit():
        m = _ORDINAL_RE.match(t)
        if m:
            # canonical ordinal suffix: noisy "126nd" / "105rd" -> "126th" / "105th"
            return _ordinal(int(m.group(1)))
        # "5ervices" style: leading digit followed by letters only
        if len(t) > 3 and t[1:].isalpha():
            return t.translate(_LEET)
    return t


def name_tokens(raw, tl=None):
    """Return (tokens_all, core_tokens, legal_set, alt_names) for a business name."""
    if not raw:
        return [], [], frozenset(), []
    if tl is not None:
        raw = tl(raw)
    s = _basic(raw)
    alts = []
    m = _DBA_RE.search(s)
    if m:
        alts = [s[: m.start()], s[m.end():]]
        s = s[: m.start()] + " " + s[m.end():]
    w = _WEB_RE.search(s)
    if w:
        s = s[: w.start()] + " " + w.group(1).replace("-", " ") + " " + s[w.end():]
    # dotted acronyms / legal forms: "s.a.s.u." "e.u.r.l." "l.l.c." -> "sasu" "eurl" "llc"
    s = _DOTTED_RE.sub(lambda m: " " + m.group(0).replace(".", "") + " ", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"\bl\.?\s?l\.?\s?c\b\.?", " llc ", s)
    s = re.sub(r"\bl\.?\s?l\.?\s?p\b\.?", " llp ", s)
    s = re.sub(r"\bp\.?\s?l\.?\s?c\b\.?", " plc ", s)
    s = re.sub(r"\bs\.?\s?a\.?\s?r\.?\s?l\b\.?", " sarl ", s)
    s = re.sub(r"\bs\.a\.s\.?", " sas ", s)
    s = re.sub(r"\bs\.a\b\.?", " sa ", s)
    s = re.sub(r"\bpvt\.?\s?ltd\b", " pvt ltd ", s)
    s = re.sub(r"\bpra\.?\s+li\b\.?", " pvt ltd ", s)
    s = s.replace("'s ", "s ").replace("'", "")
    toks = [_fix_token(t) for t in _WORD_RE.findall(s)]
    toks = [LEGAL_MAP.get(t, t) for t in toks]
    # legal-form set: genuine legal forms only (connectives / country words are
    # dropped from the core name but must not count as a shared legal form)
    legal = frozenset(t for t in toks if t in LEGAL_CORE_DROP and t not in NON_LEGAL)
    core = [t for t in toks if t not in LEGAL_CORE_DROP]
    if not core:
        core = [t for t in toks if t not in ("and", "the", "of")] or toks
    alt_core = []
    for a in alts:
        at = [LEGAL_MAP.get(_fix_token(t), _fix_token(t)) for t in _WORD_RE.findall(a.replace("'", ""))]
        at = [t for t in at if t not in LEGAL_CORE_DROP]
        if at:
            alt_core.append(at)
    return toks, core, legal, alt_core


def addr_parse(raw, tl=None):
    """Return dict with normalised tokens, numbers, state tag and components."""
    if not raw:
        return {"toks": [], "nums": [], "state": "", "comps": []}
    if tl is not None:
        raw = tl(raw, is_addr=True)
    s = _basic(raw)
    s = s.replace("n°", " ").replace("#", " ")
    comps_raw = [c.strip() for c in re.split(r"[,;]", s) if c.strip()]
    toks, nums, comps, state = [], [], [], ""
    for c in comps_raw:
        c2 = re.sub(r"[^a-z0-9/\-\s]", " ", c)
        # house-number prefixes glued to the number: "no123", "hno12", "plot7" -> "no 123"
        c2 = _GLUED_RE.sub(r"\1 \2", c2)
        c2 = re.sub(r"\s+", " ", c2).strip()
        if c2 in ("null", "n/a", "na", "none", ""):
            continue
        key = c2.replace("-", " ")
        if key in STATE_COMPONENT:
            state = STATE_COMPONENT[key]
            continue
        if key in STATE_ABBR_COMPONENT or key in IN_ABBR:
            # a stand-alone 2-letter component is a state abbreviation
            state = "ab_" + key
            continue
        key = CITY_ALIAS.get(key, key)
        ctoks = []
        for t in re.split(r"[\s/]+", key):
            t = t.strip("-")
            if not t:
                continue
            if any(ch.isdigit() for ch in t):
                # numbers: strip leading zeros in each dash-separated part
                parts = [p.lstrip("0") or "0" if p.isdigit() else _fix_token(p) for p in t.split("-") if p]
                t = "-".join(parts)
                if t:
                    nums.append(t)
                    for p in parts:
                        if p.isdigit() and p != t:
                            nums.append(p)
            else:
                t = ADDR_MAP.get(t, t)
                t = CITY_ALIAS.get(t, t)
            if t and t not in ADDR_FILLER:
                ctoks.append(t)
        if ctoks:
            comps.append(" ".join(ctoks))
            toks.extend(ctoks)
    return {"toks": toks, "nums": nums, "state": state, "comps": comps}


STATE_FULL_TO_ABBR = {}
for full, ab in US_STATES.items():
    STATE_FULL_TO_ABBR["st_" + ab] = ab
for full, ab in IN_STATES.items():
    STATE_FULL_TO_ABBR["st_in_" + ab] = ab
_IN_ALT = {"tg": ["ts"], "od": ["or"], "uk": ["ut"]}


def state_key(tag):
    """Map a parsed state tag to a comparable 2-letter-ish key."""
    if not tag:
        return ""
    if tag.startswith("ab_"):
        k = tag[3:]
    else:
        k = STATE_FULL_TO_ABBR.get(tag, tag)
    return _IN_ALT_REV.get(k, k)


_IN_ALT_REV = {"ts": "tg", "or": "od", "ut": "uk"}
