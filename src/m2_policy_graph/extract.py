"""
extract.py — Lexicon-based extractors for data types, purposes, and third parties.

All functions are rule-based (regex / keyword matching) — no heavy NER or
fine-tuning required.  They are the baseline extractors for M2 and will be
augmented/replaced in later modules.

Public API:
  extract_data_types(text)    -> list[str]   canonical DATA_TYPES labels
  extract_purposes(text)      -> list[str]   canonical PURPOSES labels
  extract_third_parties(text) -> list[str]   company names from THIRD_PARTIES
"""

from __future__ import annotations

import re
from typing import Dict, List, Pattern, Set

from m2_policy_graph.vocab import DATA_TYPES, PURPOSES, THIRD_PARTIES

# ---------------------------------------------------------------------------
# Data-type synonym table
# Each canonical data-type key maps to a list of regex patterns (case-insensitive).
# ---------------------------------------------------------------------------

_DATA_TYPE_PATTERNS: Dict[str, List[str]] = {
    "name": [
        r"\byour (?:full |legal |real )?name\b",
        r"\bfirst(?: and last)? name\b",
        r"\blast name\b",
        r"\bfull name\b",
        r"\buser ?name\b",
        r"\bscreen ?name\b",
    ],
    "email_address": [
        r"\be-?mail(?: address)?\b",
        r"\byour email\b",
        r"\bemail(?: id)?\b",
    ],
    "phone_number": [
        r"\bphone(?: number)?\b",
        r"\bmobile(?: number)?\b",
        r"\btelephone(?: number)?\b",
        r"\bcell(?: number| phone)?\b",
    ],
    "address": [
        r"\bphysical address\b",
        r"\bmailing address\b",
        r"\bpostal address\b",
        r"\bbilling address\b",
        r"\bstreet address\b",
        r"\bzip(?: code)?\b",
        r"\bpostcode\b",
    ],
    "user_ids": [
        r"\buser ?id\b",
        r"\baccount ?id\b",
        r"\bprofile ?id\b",
        r"\busername\b",
        r"\bhandle\b",
    ],
    "date_of_birth": [
        r"\bdate of birth\b",
        r"\bbirthdate\b",
        r"\bbirthday\b",
        r"\bage\b",
        r"\bdob\b",
    ],
    "race_ethnicity": [
        r"\brace\b",
        r"\bethnicity\b",
        r"\bnational origin\b",
        r"\brace and ethnicity\b",
    ],
    "political_or_religious_beliefs": [
        r"\bpolitical(?: belief| view| opinion)?\b",
        r"\breligious(?: belief| view| affiliation)?\b",
        r"\bpolitical or religious\b",
    ],
    "sexual_orientation": [
        r"\bsexual orientation\b",
        r"\bgender identity\b",
    ],
    "user_payment_info": [
        r"\bpayment(?: information| details| method| data)?\b",
        r"\bcredit card\b",
        r"\bdebit card\b",
        r"\bbank account\b",
        r"\bbilling information\b",
        r"\bfinancial information\b",
    ],
    "purchase_history": [
        r"\bpurchase(?: history)?\b",
        r"\btransaction(?: history)?\b",
        r"\border history\b",
        r"\bbuy history\b",
    ],
    "credit_score": [
        r"\bcredit score\b",
        r"\bcreditworthiness\b",
        r"\bcredit rating\b",
    ],
    "financial_info": [
        r"\bfinancial info(?:rmation)?\b",
        r"\bincome\b",
        r"\bwage\b",
        r"\bsalary\b",
        r"\bfinancial data\b",
    ],
    "precise_location": [
        r"\bGPS\b",
        r"\bgeolocation\b",
        r"\bprecise location\b",
        r"\bexact location\b",
        r"\breal-?time location\b",
        r"\blat(?:itude)? (?:and |&)?lon(?:gitude)?\b",
        r"\blocation data\b",
    ],
    "approximate_location": [
        r"\bapproximate location\b",
        r"\bcoarse location\b",
        r"\bregional location\b",
        r"\bcity-?level location\b",
        r"\bIP(?: address)? (?:to )?location\b",
        r"\bzip code location\b",
    ],
    "contacts": [
        r"\bcontacts(?: list)?\b",
        r"\baddress book\b",
        r"\bphone book\b",
        r"\bcontact(?:s)? information\b",
    ],
    "emails": [
        r"\byour emails\b",
        r"\bemail messages\b",
        r"\binbox\b",
        r"\bemail content\b",
    ],
    "sms_mms": [
        r"\bSMS\b",
        r"\bMMS\b",
        r"\btext messages?\b",
        r"\bsms messages?\b",
    ],
    "calendar_events": [
        r"\bcalendar(?: events?)?\b",
        r"\bappointments?\b",
        r"\bschedule data\b",
    ],
    "device_id": [
        r"\bdevice(?: id| identifier)?\b",
        r"\bIMEI\b",
        r"\bMAC address\b",
        r"\bhardware id\b",
        r"\bserial number\b",
    ],
    "advertising_id": [
        r"\badvertising(?: id| identifier)?\b",
        r"\bIDFA\b",
        r"\bGAID\b",
        r"\badvertisement id\b",
        r"\bad ?id\b",
        r"\bGoogle Advertising ID\b",
    ],
    "installed_apps": [
        r"\binstalled apps?\b",
        r"\binstalled applications?\b",
        r"\bapp list\b",
        r"\bpackage names?\b",
    ],
    "app_interactions": [
        r"\bapp interactions?\b",
        r"\bin-?app(?: usage| activity| behavior)?\b",
        r"\bapp usage(?: data)?\b",
        r"\bclick(?:s| behavior| data)?\b",
        r"\bnavigation data\b",
        r"\buser interactions?\b",
    ],
    "web_browsing_history": [
        r"\bweb browsing(?: history)?\b",
        r"\bbrowsing history\b",
        r"\bbrowser history\b",
        r"\bsearch history\b",
        r"\bwebsites? (?:you )?visit(?:ed)?\b",
    ],
    "in_app_search_history": [
        r"\bin-?app search(?: history)?\b",
        r"\bsearch (?:queries|terms|history) (?:within|inside) (?:the )?app\b",
    ],
    "photos": [
        r"\bphotos?\b",
        r"\bimages?\b",
        r"\bpictures?\b",
        r"\bcamera(?: roll)?\b",
        r"\bscreenshots?\b",
    ],
    "videos": [
        r"\bvideos?\b",
        r"\bvideo recordings?\b",
        r"\bvideo files?\b",
    ],
    "files_and_docs": [
        r"\bfiles?(?: and docs?)?\b",
        r"\bdocuments?\b",
        r"\bfile storage\b",
        r"\bstorage access\b",
        r"\bmedia files?\b",
    ],
    "voice_or_sound_recordings": [
        r"\bvoice recordings?\b",
        r"\baudio recordings?\b",
        r"\bmicrophone\b",
        r"\bsound recordings?\b",
        r"\bspoken (?:content|data)\b",
    ],
    "music_files": [
        r"\bmusic files?\b",
        r"\baudio files?\b",
        r"\bsongs?\b",
        r"\bmusic library\b",
    ],
    "health_info": [
        r"\bhealth(?: information| data)?\b",
        r"\bmedical(?: information| records| data)?\b",
        r"\bhealth conditions?\b",
        r"\bdiagnosis\b",
        r"\bprescription\b",
    ],
    "fitness_info": [
        r"\bfitness(?: data| information)?\b",
        r"\bworkout(?: data)?\b",
        r"\bsteps?(?: count)?\b",
        r"\bexercise data\b",
        r"\bactivity tracking\b",
    ],
    "biometric_data": [
        r"\bbiometric(?: data)?\b",
        r"\bfingerprint\b",
        r"\bface recognition\b",
        r"\biris scan\b",
        r"\bvoice print\b",
    ],
    "crash_logs": [
        r"\bcrash logs?\b",
        r"\bcrash reports?\b",
        r"\berror logs?\b",
        r"\bbug reports?\b",
    ],
    "diagnostics": [
        r"\bdiagnostic(?: data| information)?\b",
        r"\bperformance data\b",
        r"\bapp performance\b",
        r"\bsystem logs?\b",
        r"\btelemetry\b",
    ],
    "other_app_performance_data": [
        r"\bapp performance data\b",
        r"\bperformance metrics?\b",
        r"\bload times?\b",
        r"\blatency data\b",
    ],
    "other_info": [
        r"\bother information\b",
        r"\bother data\b",
        r"\bsensitive information\b",
        r"\bpersonal information\b",
        r"\bpersonal data\b",
    ],
    "other_user_generated_content": [
        r"\buser-?generated content\b",
        r"\bUGC\b",
        r"\bposts?\b",
        r"\bcomments?\b",
        r"\breviews?\b",
        r"\bfeedback\b",
    ],
}

# Pre-compile patterns
_DATA_TYPE_COMPILED: Dict[str, List[Pattern]] = {
    key: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for key, pats in _DATA_TYPE_PATTERNS.items()
}

# ---------------------------------------------------------------------------
# Stopwords for over-generic data type triggers.
# These data types (keyed by canonical name) only fire when at least
# _STOPWORD_MIN_CONTEXT_TOKENS supporting context tokens appear within
# _STOPWORD_WINDOW words of the match.
# ---------------------------------------------------------------------------

_STOPWORD_DATA_TYPES: Set[str] = {
    "other_info",           # fires on 'personal information', 'personal data', etc.
    "other_user_generated_content",  # fires on 'posts', 'comments', 'feedback'
}

# Patterns inside other_info that are truly too generic without context
_GENERIC_TRIGGERS: Set[str] = {
    "personal information", "personal data", "other information",
    "other data", "sensitive information", "information", "data",
}

_STOPWORD_CONTEXT_TOKENS: List[Pattern] = [
    re.compile(r"\bwe collect\b", re.IGNORECASE),
    re.compile(r"\bwe share\b", re.IGNORECASE),
    re.compile(r"\bwe (?:may )?(?:use|process|gather|receive|obtain|store|disclose)\b", re.IGNORECASE),
    re.compile(r"\bcollect(?:ing|ion)?\b", re.IGNORECASE),
    re.compile(r"\bshare(?:d|s|ing)?\b", re.IGNORECASE),
    re.compile(r"\bprocess(?:ing|ed)?\b", re.IGNORECASE),
    re.compile(r"\bdisclose\b", re.IGNORECASE),
    re.compile(r"\btransfer\b", re.IGNORECASE),
    re.compile(r"\bprovide(?:d|s)?\b", re.IGNORECASE),
    re.compile(r"\bsubmit\b", re.IGNORECASE),
]

_STOPWORD_MIN_CONTEXT_TOKENS = 2   # require at least 2 context hits within window
_STOPWORD_WINDOW = 10              # tokens either side of the match


def _has_enough_context(text: str, match_start: int, match_end: int) -> bool:
    """
    Return True when at least _STOPWORD_MIN_CONTEXT_TOKENS context patterns
    appear within _STOPWORD_WINDOW whitespace-delimited tokens of the match.
    """
    tokens = text.split()
    # Map character position to token index (approximation via cumulative lengths)
    char_pos = 0
    tok_starts = []
    for tok in tokens:
        tok_starts.append(char_pos)
        char_pos += len(tok) + 1  # +1 for the space

    # Find token index closest to match midpoint
    mid = (match_start + match_end) // 2
    match_tok = 0
    for i, ts in enumerate(tok_starts):
        if ts <= mid:
            match_tok = i
        else:
            break

    lo = max(0, match_tok - _STOPWORD_WINDOW)
    hi = min(len(tokens), match_tok + _STOPWORD_WINDOW + 1)
    window_text = " ".join(tokens[lo:hi])

    hits = sum(1 for cp in _STOPWORD_CONTEXT_TOKENS if cp.search(window_text))
    return hits >= _STOPWORD_MIN_CONTEXT_TOKENS

# ---------------------------------------------------------------------------
# Purpose synonym table
# ---------------------------------------------------------------------------

_PURPOSE_PATTERNS: Dict[str, List[str]] = {
    "app_functionality": [
        r"\bapp functionality\b",
        r"\bcore functionality\b",
        r"\bprovide (?:the )?service\b",
        r"\boperate (?:the )?(?:service|app|application)\b",
        r"\bto function\b",
        r"\bservice delivery\b",
        r"\bproduct features?\b",
        r"\bfunctionality\b",
    ],
    "analytics": [
        r"\banalytics?\b",
        r"\banalysis\b",
        r"\bstatistics?\b",
        r"\bmetrics?\b",
        r"\bto improve (?:our )?service\b",
        r"\bservice improvement\b",
        r"\bproduct improvement\b",
        r"\bperformance monitoring\b",
        r"\busage data\b",
        r"\busage statistics\b",
        r"\bmeasure\b",
    ],
    "advertising_marketing": [
        r"\badvertis(?:ing|ement|e)\b",
        r"\bmarketing\b",
        r"\bads?\b",
        r"\btargeted ads?\b",
        r"\bbehavioral advertising\b",
        r"\binterest-?based ads?\b",
        r"\bpromotion(?:al)?\b",
        r"\bcommercial communications?\b",
        r"\bdirect marketing\b",
    ],
    "personalization": [
        r"\bpersonali[sz]ation\b",
        r"\bpersonali[sz]e\b",
        r"\bcustomi[sz]ation\b",
        r"\bcustomi[sz]e\b",
        r"\btailor(?:ed)?(?: content| experience| recommendations?)?\b",
        r"\brecommendations?\b",
        r"\buser preferences?\b",
    ],
    "account_management": [
        r"\baccount management\b",
        r"\bmanage (?:your )?account\b",
        r"\baccount creation\b",
        r"\bregistration\b",
        r"\bauthentication\b",
        r"\bsign[\s-]?(?:in|up)\b",
        r"\blog[\s-]?in\b",
    ],
    "developer_communications": [
        r"\bdeveloper communications?\b",
        r"\bservice (?:updates|notifications)\b",
        r"\bproduct (?:updates|news)\b",
        r"\btransactional emails?\b",
        r"\bnotif(?:y|ications?) (?:you|users?)\b",
        r"\bcommunicate with you\b",
        r"\bnewsletters?\b",
    ],
    "fraud_prevention_security": [
        r"\bfraud prevention\b",
        r"\bfraud detection\b",
        r"\bsecurity\b",
        r"\bsafety\b",
        r"\bprotect(?:ion)?\b",
        r"\bcompliance\b",
        r"\blaw enforcement\b",
        r"\babuse prevention\b",
        r"\banti-?fraud\b",
        r"\bauthenticity\b",
    ],
    "compliance_legal": [
        r"\blegal (?:obligation|requirement|compliance)\b",
        r"\bregulatory (?:requirement|compliance)\b",
        r"\bGDPR\b",
        r"\bCCPA\b",
        r"\blegal basis\b",
        r"\blegal duty\b",
        r"\blegal proceeding\b",
        r"\blaw enforcement request\b",
        r"\blawful (?:basis|obligation)\b",
    ],
}

_PURPOSE_COMPILED: Dict[str, List[Pattern]] = {
    key: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for key, pats in _PURPOSE_PATTERNS.items()
}

# ---------------------------------------------------------------------------
# Third-party matching — compiled word-boundary patterns
# ---------------------------------------------------------------------------

_MIN_COMPANY_NAME_LEN = 4  # skip names shorter than this to avoid FP

# Common words that appear in THIRD_PARTIES but are too generic to match reliably
_STOP_COMPANY_NAMES: set[str] = {
    "directly", "manage", "access", "contact", "data", "mobile", "media",
    "network", "digital", "social", "connect", "platform", "services",
    "group", "system", "global", "marketing", "analytics", "creative",
    "intelligence", "interactive", "solutions", "technology", "technologies",
    "online", "partner", "partners", "performance", "audience", "audiences",
    "content", "display", "targeting", "identity", "profile", "signal",
}


def _build_third_party_patterns() -> List[tuple[str, Pattern]]:
    """
    Compile word-boundary regex patterns for each company in THIRD_PARTIES.
    Skip entries that are too short or too generic.
    """
    compiled = []
    for name in THIRD_PARTIES:
        if len(name) < _MIN_COMPANY_NAME_LEN:
            continue
        # Skip generic common words that cause false positives
        if name.lower() in _STOP_COMPANY_NAMES:
            continue
        # Escape for regex, then wrap in word boundaries
        escaped = re.escape(name)
        try:
            pat = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
            compiled.append((name, pat))
        except re.error:
            pass
    return compiled


_THIRD_PARTY_PATTERNS: List[tuple[str, Pattern]] = _build_third_party_patterns()


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------

def extract_data_types(text: str) -> List[str]:
    """
    Extract canonical DATA_TYPE labels from text using regex/keyword matching.

    For stopword data types (e.g. 'other_info') that fire on overly-generic
    terms, at least _STOPWORD_MIN_CONTEXT_TOKENS supporting context tokens must
    appear within _STOPWORD_WINDOW words of the match before the label is added.
    This reduces OVER_DISCLOSURE false positives from boilerplate policy language.

    Returns a deduplicated list of canonical label strings (e.g. 'email_address').
    """
    found: List[str] = []
    seen: Set[str] = set()

    for dtype, patterns in _DATA_TYPE_COMPILED.items():
        if dtype in seen:
            continue
        for pat in patterns:
            m = pat.search(text)
            if m:
                # Apply context gate for stopword data types
                if dtype in _STOPWORD_DATA_TYPES:
                    if not _has_enough_context(text, m.start(), m.end()):
                        continue  # try next pattern — maybe another is more specific
                found.append(dtype)
                seen.add(dtype)
                break

    return found


def extract_purposes(text: str) -> List[str]:
    """
    Extract canonical PURPOSE labels from text using regex/keyword matching.

    Returns a deduplicated list of canonical purpose strings
    (e.g. 'advertising_marketing').
    """
    found: List[str] = []
    seen: Set[str] = set()

    for purpose, patterns in _PURPOSE_COMPILED.items():
        if purpose not in seen:
            for pat in patterns:
                if pat.search(text):
                    found.append(purpose)
                    seen.add(purpose)
                    break

    return found


def extract_third_parties(text: str) -> List[str]:
    """
    Extract third-party company names from text using word-boundary matching
    against the THIRD_PARTIES vocabulary.

    Requires word boundaries and skips names shorter than 4 characters.
    Returns a deduplicated list of matched company name strings.
    """
    found: List[str] = []
    seen: Set[str] = set()

    for name, pat in _THIRD_PARTY_PATTERNS:
        norm_name = name.lower()
        if norm_name not in seen and pat.search(text):
            found.append(name)
            seen.add(norm_name)

    return found


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = (
        "We collect your email address and phone number to provide app functionality "
        "and for analytics. We may share your GPS location with Google and Facebook "
        "for advertising and marketing purposes. Crash logs are used for diagnostics."
    )
    print("Text:", demo[:80], "...")
    print("Data types:", extract_data_types(demo))
    print("Purposes:", extract_purposes(demo))
    print("Third parties:", extract_third_parties(demo))
