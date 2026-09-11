# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request

try:
    from pypinyin import lazy_pinyin, Style
    HAS_PYPINYIN = True
except Exception:
    HAS_PYPINYIN = False
    lazy_pinyin = None
    Style = None

MODEL = "tencent/HY-MT1.5-1.8B-GGUF:Q4_K_M"
PORT = 18087
CONTEXT = 3072
STARTUP_TIMEOUT = 1800
SEED = 42
STOP = ["<ï½œhy_placeâ–holderâ–noâ–2ï½œ>", "<ï½œhy_Userï½œ>"]

HERE = Path(__file__).resolve().parent
KB = HERE / "knowledge-base"


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


def http_json(url: str, payload=None, timeout=15):
    data = None
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw.decode("utf-8")) if raw else {}


def ready(base: str) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(base + path, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            pass
    return False


def find_server():
    candidates = []
    env = os.environ.get("LLAMA_SERVER_EXE", "").strip()
    if env:
        candidates.append(env)
    candidates += [
        str(HERE / "llama-server.exe"),
        str(HERE / "llama-server" / "llama-server.exe"),
    ]
    for name in ("llama-server.exe", "llama-server"):
        hit = shutil.which(name)
        if hit:
            candidates.append(hit)
    for item in candidates:
        if Path(item).is_file():
            return str(Path(item))
    return None


def start_server(exe: str):
    cmd = [
        exe, "-hf", MODEL,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-c", str(CONTEXT),
        "-ngl", "0",
        "--jinja",
    ]
    threads = os.environ.get("HYMT_THREADS", "").strip()
    if threads.isdigit() and int(threads) > 0:
        cmd += ["-t", threads]

    log = (HERE / "hymt15-xianxia-FIX649A-server.log").open(
        "w", encoding="utf-8", errors="replace"
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        cwd=str(HERE),
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    return proc, log, cmd


def wait_server(base: str, proc):
    started = time.time()
    next_notice = started
    while time.time() - started < STARTUP_TIMEOUT:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server thoÃ¡t sá»›m code={proc.returncode}; xem hymt15-xianxia-FIX649A-server.log"
            )
        if ready(base):
            return
        now = time.time()
        if now >= next_notice:
            print(f"[WAIT] model server {int(now-started)}s...", flush=True)
            next_notice = now + 15
        time.sleep(2)
    raise TimeoutError("llama-server khÃ´ng sáºµn sÃ ng")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_terms():
    term_files = [
        KB / "common" / "semantic-core.json",
        KB / "xianxia" / "roles-titles.json",
        KB / "xianxia" / "cultivation-terms.json",
        KB / "xianxia" / "worlds-factions.json",
        KB / "xianxia" / "items-artifacts.json",
        KB / "xianxia" / "domain-ontology-core.json",
        KB / "xianxia" / "beasts-races.json",
        KB / "xianxia" / "idioms-fixed-expressions.json",
    ]
    rows = []
    for path in term_files:
        data = load_json(path)
        for entry in data.get("entries", []):
            row = dict(entry)
            row["_pack"] = data.get("packId", path.name)
            rows.append(row)
    return rows


def load_surnames():
    data = load_json(KB / "common" / "chinese-surnames-v2.json")
    rows = []
    for entry in data.get("entries", []):
        row = dict(entry)
        row["_pack"] = data.get("packId", "surnames")
        rows.append(row)
    return rows


def load_grammar():
    data = load_json(KB / "common" / "chinese-grammar-segmentation.json")
    return {k: list(v) for k, v in data.get("categories", {}).items()}


def is_cjk(ch: str) -> bool:
    return bool(ch) and "\u3400" <= ch <= "\u9fff"


def cjk_count(text: str):
    return sum(1 for ch in text if is_cjk(ch))


def pinyin_key(text: str):
    if not HAS_PYPINYIN:
        return None
    try:
        return " ".join(
            lazy_pinyin(
                text,
                style=Style.TONE3,
                neutral_tone_with_five=True,
                errors="default",
            )
        )
    except Exception:
        return None


def hamming_chars(a: str, b: str):
    if len(a) != len(b):
        return 999
    return sum(x != y for x, y in zip(a, b))


def build_alias_index(terms):
    index = []
    for entry in terms:
        forms = [entry.get("source", "")] + list(entry.get("aliases") or [])
        forms = [x for x in forms if x]
        index.append((entry, forms))
    index.sort(key=lambda item: max(len(x) for x in item[1]), reverse=True)
    return index


def retrieve_terms(source: str, alias_index, max_terms=6):
    """Longest non-overlapping knowledge retrieval.

    A longer exact domain term owns its covered source region so nested entries do not
    create contradictory guidance/QA (e.g. a formation-device term containing a shorter
    formation term). A shorter term can still be selected if it also occurs elsewhere.
    """
    candidates = []
    for entry, forms in alias_index:
        occ = []
        for form in forms:
            if not form:
                continue
            start = 0
            while True:
                pos = source.find(form, start)
                if pos < 0:
                    break
                occ.append((pos, pos + len(form), form))
                start = pos + len(form)
        if occ:
            candidates.append((entry, occ))
    candidates.sort(key=lambda item: max((b-a) for a,b,_ in item[1]), reverse=True)

    selected, covered, seen_src = [], [], set()
    for entry, occ in candidates:
        src = entry.get("source", "")
        if src in seen_src:
            continue
        usable = []
        for a,b,form in occ:
            if any(a >= ca and b <= cb for ca,cb in covered):
                continue
            usable.append((a,b,form))
        if not usable:
            continue
        selected.append(entry); seen_src.add(src)
        # Claim the longest uncovered occurrences for containment suppression.
        for a,b,form in usable:
            covered.append((a,b))
        if len(selected) >= max_terms:
            break
    return selected

def detect_honorifics(source: str, terms):
    return [
        e for e in terms
        if e.get("category") in {"honorific", "relationship-title", "sect-role"}
        and e.get("source") and e.get("source") in source
    ]


def preceding_cjk_token(source: str, pos: int, max_len=4):
    i = pos - 1
    chars = []
    while i >= 0 and is_cjk(source[i]) and len(chars) < max_len:
        chars.append(source[i])
        i -= 1
    return "".join(reversed(chars))


def build_surname_maps(surnames):
    exact = {}
    pinyin2 = {}
    for e in surnames:
        src = e.get("source", "")
        if not src:
            continue
        exact[src] = e
        if HAS_PYPINYIN and len(src) == 2:
            key = pinyin_key(src)
            if key:
                pinyin2.setdefault(key, []).append(e)
    return exact, pinyin2


def detect_names(source, title_terms, surname_exact, surname_pinyin2, grammar=None):
    """Data-driven Chinese name detector with grammatical boundary protection.

    Exact surnames remain strong evidence, but an optional given-name suffix may not cross
    a function-word/predicate boundary before a title. This prevents false parses such as
    ...æ³•å°æ˜¯å¸ˆçˆ¶... -> surname å° + given-name æ˜¯.
    """
    names = []
    homophones = []
    grammar = grammar or {}
    name_boundary_tokens = set()
    for cat in (
        "common_verbs", "structural_particles", "negation", "modality",
        "aspect_time", "conjunctions", "pronouns"
    ):
        name_boundary_tokens.update(grammar.get(cat, []))
    name_boundary_tokens.update({"æ˜¯", "ä¸º", "ç»™", "æŠŠ", "è¢«", "å‘", "å¯¹", "ä»Ž", "åœ¨", "ä¸Ž", "å’Œ"})
    non_name_lexemes = set(grammar.get("non_name_lexemes", []))

    title_forms = []
    for e in title_terms:
        forms = [e.get("source", "")] + list(e.get("aliases") or [])
        for form in forms:
            if form and form in source:
                title_forms.append((form, e))
    title_forms.sort(key=lambda x: len(x[0]), reverse=True)
    surname_items = sorted(surname_exact.items(), key=lambda kv: len(kv[0]), reverse=True)

    for title, h in title_forms:
        start = 0
        while True:
            pos = source.find(title, start)
            if pos < 0:
                break
            start = pos + max(1, len(title))
            exact_entry = None
            exact_surface = None
            name_surface = None
            left = source[max(0, pos - 4):pos]
            for surname, entry in surname_items:
                rel = left.rfind(surname)
                if rel < 0:
                    continue
                # A one-character surname candidate must not be the tail of an ordinary
                # lexical item immediately before a title, e.g. ä»»ä½•å¼Ÿå­ / æŽ¢è·¯å¼Ÿå­.
                suffix = left[rel + len(surname):]
                candidate_surface = surname + suffix
                if len(surname) == 1 and (
                    candidate_surface in non_name_lexemes
                    or any(lex and len(lex) > 1 and left.endswith(lex) and (lex.endswith(surname) or lex == candidate_surface) for lex in non_name_lexemes)
                ):
                    continue
                suffix_bad = False
                if suffix:
                    for b in sorted(name_boundary_tokens, key=len, reverse=True):
                        if b and (suffix == b or suffix.endswith(b)):
                            suffix_bad = True
                            break
                if len(suffix) <= 2 and all(is_cjk(ch) for ch in suffix) and not suffix_bad:
                    exact_entry = entry
                    exact_surface = surname
                    name_surface = left[rel:]
                    break

            if exact_entry:
                names.append({
                    "nameSource": name_surface or exact_surface,
                    "titleSource": title,
                    "surnameSurface": exact_surface,
                    "surnameCandidate": exact_entry.get("source"),
                    "surnameVietnamese": exact_entry.get("preferredTarget"),
                    "surnameEvidence": "EXACT_COMMON_SURNAME",
                    "authority": "SOFT",
                })
                continue

            if HAS_PYPINYIN:
                immediate = source[max(0, pos - 2):pos]
                if len(immediate) == 2 and all(is_cjk(ch) for ch in immediate):
                    # Do not create a homophone name risk across the same boundary class.
                    if any(immediate.endswith(b) for b in name_boundary_tokens if b):
                        continue
                    key = pinyin_key(immediate)
                    for e in surname_pinyin2.get(key, []):
                        known = e.get("source", "")
                        if known == immediate:
                            continue
                        if hamming_chars(immediate, known) <= 1:
                            risk = {
                                "sourceToken": immediate,
                                "candidateSurname": known,
                                "candidateVietnamese": e.get("preferredTarget", ""),
                                "pinyin": key,
                                "nameContext": immediate,
                                "titleSource": title,
                                "authority": "ADVISORY_ONLY",
                            }
                            homophones.append(risk)
                            names.append({
                                "nameSource": immediate,
                                "titleSource": title,
                                "surnameSurface": immediate,
                                "surnameCandidate": known,
                                "surnameVietnamese": e.get("preferredTarget", ""),
                                "surnameEvidence": "HOMOPHONE_COMPOUND_SURNAME_RISK",
                                "authority": "ADVISORY_ONLY",
                            })
                            break

    def dedupe(rows, keys):
        out, seen = [], set()
        for r in rows:
            k = tuple(r.get(x) for x in keys)
            if k not in seen:
                seen.add(k); out.append(r)
        return out

    return (
        dedupe(names, ["nameSource", "titleSource", "surnameCandidate"]),
        dedupe(homophones, ["sourceToken", "candidateSurname"]),
    )

def make_segmentation_lexicon(terms, surnames, grammar, names, sense_rules=None):
    words = set()

    for e in terms:
        s = e.get("source", "")
        if s:
            words.add(s)
        for a in e.get("aliases") or []:
            if a:
                words.add(a)

    for e in surnames:
        s = e.get("source", "")
        if s:
            words.add(s)

    # Contextual-sense entries contribute ONLY lexical boundaries, not target authority.
    for e in sense_rules or []:
        src = e.get("source", "")
        if src:
            words.add(src)
        for a in e.get("aliases") or []:
            if a:
                words.add(a)

    for vals in grammar.values():
        words.update(x for x in vals if x)

    for n in names:
        if n.get("nameSource"):
            words.add(n["nameSource"])

    return words

def segment_chinese(source: str, lexicon):
    max_len = max([len(x) for x in lexicon] + [1])
    out = []
    i = 0

    while i < len(source):
        ch = source[i]

        if not is_cjk(ch):
            if not ch.isspace():
                out.append(ch)
            i += 1
            continue

        match = None
        for size in range(min(max_len, len(source) - i), 0, -1):
            cand = source[i:i+size]
            if cand in lexicon:
                match = cand
                break

        if match is None:
            match = ch

        out.append(match)
        i += len(match)

    return out


def token_hits(tokens, vocabulary):
    vocab = set(vocabulary)
    return [t for t in tokens if t in vocab]


def _token_offsets(source, tokens):
    out = []
    cursor = 0
    for tok in tokens:
        pos = source.find(tok, cursor)
        if pos < 0:
            pos = cursor
        out.append((tok, pos, pos + len(tok)))
        cursor = pos + len(tok)
    return out


def detect_polar_questions(source, special_policy=None):
    rows = []
    for pat in sorted(set((special_policy or {}).get("polarQuestionPatterns", [])), key=len, reverse=True):
        start = 0
        while pat:
            pos = source.find(pat, start)
            if pos < 0:
                break
            rows.append({"sourceSpan": pat, "start": pos, "end": pos + len(pat), "role": "POLAR_QUESTION"})
            start = pos + len(pat)
    for m in re.finditer(r"([\u3400-\u9fff])ä¸\1", source):
        rows.append({"sourceSpan": m.group(0), "start": m.start(), "end": m.end(), "role": "POLAR_QUESTION"})
    best = {(r["start"], r["end"]): r for r in rows}
    return sorted(best.values(), key=lambda r: (r["start"], -(r["end"]-r["start"])))


def _polarity_suppression_ranges(source, special_policy, polar_rows):
    rows = [(x["start"], x["end"], "polar-question") for x in polar_rows]
    for phrase in (special_policy or {}).get("negationLexicalizedNonNegative", []):
        start = 0
        while phrase:
            pos = source.find(phrase, start)
            if pos < 0:
                break
            rows.append((pos, pos + len(phrase), f"lexical:{phrase}"))
            start = pos + len(phrase)
    for neg, rights in ((special_policy or {}).get("negationLexicalExceptions", {}) or {}).items():
        for right in rights or []:
            phrase = neg + right
            start = 0
            while phrase:
                pos = source.find(phrase, start)
                if pos < 0:
                    break
                rows.append((pos, pos + len(phrase), f"lexical:{phrase}"))
                start = pos + len(phrase)
    return rows

def analyze_grammar(source, tokens, grammar, special_policy=None):
    pronouns = token_hits(tokens, grammar.get("pronouns", []))
    neg_vocab = set(grammar.get("negation", []))
    polar_questions = detect_polar_questions(source, special_policy)
    suppress = _polarity_suppression_ranges(source, special_policy, polar_questions)
    negation = []
    neg_occ = []
    for tok, st, en in _token_offsets(source, tokens):
        if tok not in neg_vocab:
            continue
        if any(st >= a and en <= b for a, b, _ in suppress):
            continue
        negation.append(tok)
        neg_occ.append({"source": tok, "start": st, "end": en})

    modality = token_hits(tokens, grammar.get("modality", []))
    aspect_time = token_hits(tokens, grammar.get("aspect_time", []))
    conjunctions = token_hits(tokens, grammar.get("conjunctions", []))
    particles = token_hits(tokens, grammar.get("structural_particles", []))
    features, expectation = [], []
    if pronouns: features.append("reference_or_pronoun")
    if negation: features.append("negation")
    if polar_questions: features.append("polar_question")
    if modality: features.append("modality")
    if aspect_time: features.append("aspect_or_time")
    if conjunctions: features.append("discourse_connector")
    if source.endswith(("çš„", "ä¹‹")):
        features.append("fragment_or_nominal_ellipsis"); expectation.append("expects_following_or_omitted_head")
    if source.startswith(("ä¹Ÿ", "åˆ", "è¿˜", "å´", "ä½†", "è€Œ", "æ‰€ä»¥", "é‚£ä¹ˆ")):
        features.append("continuation_from_previous"); expectation.append("inherits_previous_proposition_or_topic")
    if source in set(grammar.get("response_fragments", [])):
        features.append("short_response"); expectation.append("meaning_depends_on_previous_utterance")
    pronoun_vocab = set(grammar.get("pronouns", [])); particle_vocab = set(grammar.get("structural_particles", []))
    if bool(tokens) and all(t in pronoun_vocab | particle_vocab for t in tokens):
        features.append("pronoun_fragment"); expectation.append("expects_predicate_or_discourse_context")
    if cjk_count(source) <= 4: features.append("short_subtitle_fragment")
    if any(x in source for x in ("ä»–","å¥¹","å®ƒ","ä»–ä»¬","å¥¹ä»¬","å®ƒä»¬","è¿™","é‚£","å…¶")):
        expectation.append("may_need_antecedent_or_referent")
    context_needed = any(f in features for f in (
        "fragment_or_nominal_ellipsis", "continuation_from_previous", "short_response",
        "pronoun_fragment", "short_subtitle_fragment", "reference_or_pronoun"))
    return {
        "tokens": tokens, "pronouns": pronouns, "negation": negation,
        "negationOccurrences": neg_occ, "polarQuestions": polar_questions,
        "modality": modality, "aspectTime": aspect_time, "conjunctions": conjunctions,
        "particles": particles, "features": list(dict.fromkeys(features)),
        "nextSemanticExpectation": list(dict.fromkeys(expectation)), "contextNeeded": context_needed,
    }

def load_sino_morphemes():
    data = load_json(KB / "common" / "sino-vietnamese-morphemes.json")
    return {e["source"]: e for e in data.get("entries", [])}


def load_compound_policy():
    return load_json(KB / "xianxia" / "sino-vietnamese-compound-patterns.json")


def load_classical_style():
    return load_json(KB / "xianxia" / "classical-vietnamese-style-rules-v2.json")


def load_contextual_senses():
    data = load_json(KB / "xianxia" / "contextual-sense-rules-v1.json")
    return list(data.get("entries", []))


def load_special_span_policy():
    return load_json(KB / "xianxia" / "special-span-policy.json")

def load_noun_phrase_policy():
    return load_json(KB / "common" / "noun-phrase-semantic-policy.json")

def load_creature_policy():
    return load_json(KB / "xianxia" / "creature-semantic-policy.json")


def _cue_hit(text: str, cues):
    return [c for c in (cues or []) if c and c in text]


def resolve_contextual_senses(source, prev_text, next_text, tokens, sense_rules):
    """
    Generic lexical-sense reranker.
    - never rewrites source;
    - rules are term/cue level, never full benchmark sentences;
    - exact current-source evidence dominates genre default;
    - context can rerank but cannot invent a lexical item absent from source.
    """
    out = []
    for entry in sense_rules:
        src = entry.get("source", "")
        aliases = [x for x in entry.get("aliases", []) if x]
        surfaces = [src] + aliases
        surface = next((x for x in surfaces if x and x in source), None)
        if not surface:
            continue

        pos = source.find(surface)
        window = int(entry.get("cueWindowChars", 8))
        left = source[max(0, pos-window):pos]
        right = source[pos+len(surface):pos+len(surface)+window]
        local = source[max(0, pos-window):pos+len(surface)+window]

        candidates = []
        for sense in entry.get("senses", []):
            score = float(sense.get("baseScore", 0.0))
            evidence = []
            when = sense.get("when") or {}

            hits = _cue_hit(left, when.get("leftAny"))
            if hits:
                score += float(when.get("leftWeight", 2.0))
                evidence += [f"left:{x}" for x in hits]

            hits = _cue_hit(right, when.get("rightAny"))
            if hits:
                score += float(when.get("rightWeight", 2.0))
                evidence += [f"right:{x}" for x in hits]

            hits = _cue_hit(local, when.get("localAny"))
            if hits:
                score += float(when.get("localWeight", 1.5))
                evidence += [f"local:{x}" for x in hits]

            hits = _cue_hit(prev_text, when.get("prevAny"))
            if hits:
                score += float(when.get("contextWeight", 0.6))
                evidence += [f"prev:{x}" for x in hits]

            hits = _cue_hit(next_text, when.get("nextAny"))
            if hits:
                score += float(when.get("contextWeight", 0.6))
                evidence += [f"next:{x}" for x in hits]

            if sense.get("genreDefault"):
                score += float(sense.get("genreDefaultWeight", 0.5))
                evidence.append("genre-default")

            candidates.append((score, sense, evidence))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        score, sense, evidence = candidates[0]
        threshold = float(entry.get("selectionThreshold", 0.5))
        if score < threshold:
            continue

        base_authority = str(sense.get("qaAuthority", "SOFT")).upper()
        lexical_evidence = [e for e in evidence if e != "genre-default"]
        # Generic confidence escalation: a SOFT lexical sense becomes STRONG only when
        # current/local lexical evidence is decisive. Genre default alone can never escalate.
        effective_authority = base_authority
        if base_authority == "SOFT" and lexical_evidence and score >= 1.8:
            effective_authority = "STRONG"

        out.append({
            "source": src,
            "surface": surface,
            "senseId": sense.get("senseId"),
            "semanticRole": sense.get("semanticRole"),
            "preferredTarget": sense.get("preferredTarget", ""),
            "targetCandidates": list(sense.get("targetCandidates") or []),
            "compoundReading": sense.get("compoundReading", ""),
            "instruction": sense.get("instruction", ""),
            "baseQaAuthority": base_authority,
            "qaAuthority": effective_authority,
            "authorityEscalated": effective_authority != base_authority,
            "forbiddenTargetPatterns": list(sense.get("forbiddenTargetPatterns") or []),
            "score": round(score, 3),
            "evidence": evidence,
            "sourceMutation": False,
        })

    return out


def detect_structural_semantic_frames(source, tokens, analysis, sino_compounds=None):
    """Reusable structural semantic frames; no benchmark sentence mapping."""
    out = []
    sino_compounds = sino_compounds or []
    strong_formation_compound = any(
        c.get("isPrimary") and str(c.get("authority", "SOFT")).upper() == "STRONG" and c.get("anchor") == "é˜µ"
        for c in sino_compounds
    )
    formation_cues = {"æˆ", "æˆå½¢", "å½¢æˆ", "å¸ƒ", "å¸ƒä¸‹", "å¸ƒç½®", "ç»“", "ç»“æˆ", "èµ·", "ç ´", "æ¯", "å´©", "ç¨³", "é•‡"}
    if not strong_formation_compound and "é˜µ" in tokens:
        for idx, tok in enumerate(tokens):
            if tok != "é˜µ": continue
            hits = sorted(set(tokens[max(0, idx-2):min(len(tokens), idx+4)]) & formation_cues)
            if hits:
                out.append({
                    "source":"é˜µ","surface":"é˜µ","senseId":"formation-structure","semanticRole":"FORMATION",
                    "preferredTarget":"tráº­n phÃ¡p","targetCandidates":["tráº­n phÃ¡p","tráº­n"],"compoundReading":"Tráº­n",
                    "instruction":"é˜µ trong cáº¥u trÃºc bá»‘ trÃ­/hÃ¬nh thÃ nh/phÃ¡ tráº­n lÃ  formation/tráº­n phÃ¡p, khÃ´ng pháº£i tráº­n chiáº¿n.",
                    "baseQaAuthority":"STRONG","qaAuthority":"STRONG","authorityEscalated":False,
                    "forbiddenTargetPatterns":["tráº­n chiáº¿n"],"score":round(2.0+0.2*len(hits),3),
                    "evidence":[f"formation-cue:{x}" for x in hits],"sourceMutation":False,"structuralFrame":True})
                break

    # Locative + å°/å°å­˜/å°å° + ç€ + entity: preserve containment direction.
    locative_forms = ("é‡Œ", "ä¸­", "å†…", "é‡Œé¢", "å†…éƒ¨", "ä¸‹é¢", "ä¸‹æ–¹", "åº•ä¸‹", "åŽé¢", "å‰é¢")
    seal_match = re.search(r"å°(?:å­˜|å°)?ç€", source)
    if seal_match and any(loc in source[:seal_match.start()] for loc in locative_forms):
        surface = seal_match.group(0)
        out.append({
            "source":surface,"surface":surface,"senseId":"sealed-containment","semanticRole":"CONTAINMENT_SEAL",
            "preferredTarget":"phong giá»¯","targetCandidates":["phong giá»¯","bá»‹ phong","Ä‘Æ°á»£c phong","phong áº¥n","niÃªm phong","giam giá»¯"],
            "compoundReading":"","instruction":"Vá»‹ trÃ­ + å°ç€ + váº­t: váº­t phÃ­a sau å°ç€ lÃ  thá»© bá»‹ phong/niÃªm giá»¯ táº¡i vá»‹ trÃ­ Ä‘Ã³; khÃ´ng Ä‘áº£o thÃ nh vá»‹ trÃ­ bá»‹ váº­t bao bá»c.",
            "baseQaAuthority":"STRONG","qaAuthority":"STRONG","authorityEscalated":False,
            "forbiddenTargetPatterns":["Ä‘Æ°á»£c bá»c bá»Ÿi","bá»‹ bá»c bá»Ÿi","Ä‘Æ°á»£c bao bá»Ÿi","bá»‹ bao bá»Ÿi"],
            "score":2.6,"evidence":["frame:LOCATIVE+SEAL+ç€"],"sourceMutation":False,"structuralFrame":True})

    # å€Ÿ direction only when syntax supplies lender/borrower evidence.
    if "å€Ÿ" in source:
        if re.search(r"é—®.{0,6}æ„¿ä¸æ„¿æ„å€Ÿ", source) or "å€Ÿç»™" in source:
            out.append({
                "source":"å€Ÿ","surface":"å€Ÿ","senseId":"lend-direction","semanticRole":"TRANSFER_LEND",
                "preferredTarget":"cho mÆ°á»£n","targetCandidates":["cho mÆ°á»£n","cho phÃ©p mÆ°á»£n","Ä‘á»“ng Ã½ cho mÆ°á»£n"],
                "compoundReading":"","instruction":"é—®æŸäººæ„¿ä¸æ„¿æ„å€Ÿ / å€Ÿç»™: ngÆ°á»i Ä‘Ã³ lÃ  bÃªn cho mÆ°á»£n; khÃ´ng Ä‘áº£o thÃ nh há» Ä‘i mÆ°á»£n.",
                "baseQaAuthority":"STRONG","qaAuthority":"STRONG","authorityEscalated":False,
                "forbiddenTargetPatterns":[],"score":2.5,"evidence":["frame:LEND_DIRECTION"],"sourceMutation":False,"structuralFrame":True})
        elif re.search(r"(?:å‘|ä»Ž).{0,6}å€Ÿ", source):
            out.append({
                "source":"å€Ÿ","surface":"å€Ÿ","senseId":"borrow-direction","semanticRole":"TRANSFER_BORROW",
                "preferredTarget":"mÆ°á»£n","targetCandidates":["mÆ°á»£n","Ä‘i mÆ°á»£n"],"compoundReading":"",
                "instruction":"å‘/ä»Ž + ngÆ°á»i + å€Ÿ biá»ƒu thá»‹ Ä‘i mÆ°á»£n tá»« ngÆ°á»i Ä‘Ã³.",
                "baseQaAuthority":"STRONG","qaAuthority":"STRONG","authorityEscalated":False,
                "forbiddenTargetPatterns":[],"score":2.4,"evidence":["frame:BORROW_DIRECTION"],"sourceMutation":False,"structuralFrame":True})

    if "ä¸æ˜¯" in source and "æ²¡æœ‰" in source and source.find("ä¸æ˜¯") < source.rfind("æ²¡æœ‰"):
        out.append({
            "source":"ä¸æ˜¯â€¦æ²¡æœ‰","surface":"ä¸æ˜¯â€¦æ²¡æœ‰","senseId":"double-negation-existence","semanticRole":"POSITIVE_RESIDUAL_EXISTENCE",
            "preferredTarget":"khÃ´ng pháº£i lÃ  khÃ´ng cÃ³","targetCandidates":["khÃ´ng pháº£i lÃ  khÃ´ng cÃ³","khÃ´ng háº³n lÃ  khÃ´ng cÃ³","váº«n cÃ³","cÅ©ng cÃ³"],
            "compoundReading":"","instruction":"ä¸æ˜¯â€¦æ²¡æœ‰ lÃ  phá»§ Ä‘á»‹nh kÃ©p: váº«n tá»“n táº¡i kháº£ nÄƒng/phÆ°Æ¡ng Ã¡n; khÃ´ng rÃºt gá»n thÃ nh 'khÃ´ng cÃ³'.",
            "baseQaAuthority":"STRONG","qaAuthority":"STRONG","authorityEscalated":False,"forbiddenTargetPatterns":[],
            "score":2.5,"evidence":["frame:DOUBLE_NEGATION_EXISTENCE"],"sourceMutation":False,"structuralFrame":True})
    return out

def merge_senses(*groups):
    out, seen = [], set()
    for group in groups:
        for s in group or []:
            key = (s.get("surface"), s.get("senseId"))
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out


def sense_reading_map(selected_senses):
    out = {}
    for s in selected_senses:
        reading = s.get("compoundReading") or s.get("preferredTarget")
        if reading:
            out[s.get("surface") or s.get("source")] = title_reading(str(reading))
    return out


def title_reading(text: str) -> str:
    # Preserve diacritics; title-case each lexical chunk for named xianxia compounds.
    return " ".join(part[:1].upper() + part[1:] if part else part for part in text.split())


def morpheme_reading(token: str, morphemes, matched_terms, selected_senses=None):
    # Contextual sense reading wins only when the source term itself was selected.
    for s in selected_senses or []:
        if (s.get("surface") == token or s.get("source") == token):
            reading = s.get("compoundReading") or s.get("preferredTarget")
            if reading:
                return title_reading(str(reading))

    # Prefer an already-retrieved genre term if it exactly covers this token.
    for e in matched_terms:
        if e.get("source") == token and e.get("preferredTarget"):
            return title_reading(str(e["preferredTarget"]))

    e = morphemes.get(token)
    if e:
        return title_reading(str(e.get("reading", "")))

    # Fall back to per-character compositional reading only if ALL chars are known.
    pieces = []
    for ch in token:
        ce = morphemes.get(ch)
        if not ce:
            return None
        pieces.append(title_reading(str(ce.get("reading", ""))))

    return " ".join(x for x in pieces if x) or None


def build_boundary_set(grammar, compound_policy):
    out = set(compound_policy.get("boundaryTokens", []))
    # Classifiers are contextual in FIX6.4.7; ä½ in å½’ä½å›¾ must not be a global boundary.
    for cat in (
        "pronouns", "negation", "modality", "aspect_time", "conjunctions",
        "structural_particles", "common_verbs"
    ):
        out.update(grammar.get(cat, []))
    return out

def _is_classifier_boundary(tokens, index, policy):
    tok = tokens[index]
    if tok not in set(policy.get("classifierBoundaryTokens", [])):
        return False
    left = tokens[index - 1] if index > 0 else ""
    if left in set(policy.get("classifierLeftQuantityCues", [])):
        return True
    if left and all(ch in "0123456789ã€‡é›¶ä¸€äºŒä¸¤ä¸‰å››äº”å…­ä¸ƒå…«ä¹åç™¾åƒä¸‡å‡ æ•°åŠ" for ch in left):
        return True
    return False


def _is_repeated_anchor_verb(tokens, index, anchor, policy):
    if index <= 0 or tokens[index - 1] != tokens[index]:
        return False
    right = tokens[index + 1] if index + 1 < len(tokens) else ""
    return right in set(policy.get("repeatedAnchorVerbRightCues", []))

def _is_element_enumeration(tokens, anchor_index, policy):
    enum = set(policy.get("enumerationTokens", []))
    if not enum or tokens[anchor_index] not in enum:
        return False
    left = anchor_index
    right = anchor_index
    while left - 1 >= 0 and tokens[left - 1] in enum:
        left -= 1
    while right + 1 < len(tokens) and tokens[right + 1] in enum:
        right += 1
    return (right - left + 1) >= int(policy.get("enumerationMinRun", 3))


def _compound_authority(span, cls, parts, policy, morphemes=None):
    strong_min = int(policy.get("strongCompoundMinChars", 3))
    strong_classes = set(policy.get("strongCompoundClasses", []))
    if cjk_count(span) >= strong_min and cls in strong_classes:
        return "STRONG"

    # Evidence-based 2-char domain compounds: strength comes from ontology category,
    # not an exact full-phrase mapping. Ordinary 2-char objects remain SOFT.
    if cjk_count(span) == 2 and cls in set(policy.get("strongTwoCharAnchorClasses", [])):
        first = parts[0] if parts else ""
        entry = (morphemes or {}).get(first, {})
        if entry.get("category") in set(policy.get("strongTwoCharModifierCategories", [])):
            return "STRONG"
    return "SOFT"



def _exact_matched_entry_for_span(span, matched_terms):
    for e in matched_terms or []:
        forms = [e.get("source", "")] + list(e.get("aliases") or [])
        if span in forms:
            return e
    return None


def _suppress_anchor_by_role(anchor, token_index, tokens, policy):
    for rule in policy.get("anchorRoleGuards", []):
        if rule.get("anchor") != anchor:
            continue
        right = tokens[token_index + 1] if token_index + 1 < len(tokens) else ""
        if right and right in set(rule.get("suppressWhenRightAny", [])):
            return True
    return False

def primary_sino_compounds(rows):
    """
    Keep only maximal overlapping compounds for prompt/QA.
    Child spans remain in the report as graph evidence, but do not compete
    with the longest structurally valid phrase.
    """
    out = []
    for row in rows:
        if not row.get("isPrimary", False):
            continue
        out.append(row)
    return out


def detect_sino_compounds(source, tokens, matched_terms, morphemes, grammar, compound_policy, selected_senses=None):
    """
    FIX6.4 multi-token resolver.

    Generic constraints:
    - anchor-driven, not sentence-driven;
    - classifier/function-word boundaries cannot enter compounds;
    - known morphemes/terms/senses compose into a phrase graph;
    - nested child spans are retained only as evidence; the maximal span is primary;
    - 3+ char xianxia-domain compounds can receive STRONG composition authority;
    - ordinary 2-char objects remain SOFT;
    - source text is never mutated.
    """
    anchor_class = {}
    for cls, vals in compound_policy.get("strongAnchors", {}).items():
        for x in vals:
            anchor_class[x] = cls

    boundary = build_boundary_set(grammar, compound_policy)
    max_chars = int(compound_policy.get("maxCompoundChars", 8))
    min_chars = int(compound_policy.get("minCompoundChars", 2))

    def resolve_anchor(tok):
        if tok in anchor_class:
            return tok, anchor_class[tok]
        # Segmentation may keep a known multi-char lexical unit such as ç¥žå‰‘/å‰‘è¯€.
        # If it ends with a strong semantic head, retain the whole token while
        # assigning the suffix as the head. Longest suffix wins.
        suffixes = [
            a for a in anchor_class
            if a and len(a) < len(tok) and tok.endswith(a)
        ]
        if not suffixes:
            return None, None
        anchor = max(suffixes, key=len)
        return anchor, anchor_class[anchor]

    non_nominal_sense_surfaces = {
        s.get("surface") or s.get("source")
        for s in (selected_senses or [])
        if s.get("semanticRole") in {"VERB_ACTION", "PHRASAL_VERB", "SPATIAL", "CONTAINMENT_SEAL"}
    }

    out = []
    for i, tok in enumerate(tokens):
        # A token already resolved as a verb/phrasal/spatial sense cannot simultaneously
        # become the semantic head of an invented named HÃ¡n-Viá»‡t compound.
        if tok in non_nominal_sense_surfaces:
            continue
        anchor, cls = resolve_anchor(tok)
        if not anchor:
            continue
        if _suppress_anchor_by_role(anchor, i, tokens, compound_policy):
            continue
        if _is_repeated_anchor_verb(tokens, i, anchor, compound_policy):
            continue

        # Generic suppression for enumerations such as é‡‘ æœ¨ æ°´ ç« åœŸ.
        if tok == anchor and _is_element_enumeration(tokens, i, compound_policy):
            continue

        parts = [tok]
        start_i = i
        j = i - 1

        while j >= 0:
            prev = tokens[j]
            if prev in boundary or _is_classifier_boundary(tokens, j, compound_policy):
                break
            if not all(is_cjk(ch) for ch in prev):
                break

            tentative = "".join([prev] + parts)
            if cjk_count(tentative) > max_chars:
                break

            if morpheme_reading(prev, morphemes, matched_terms, selected_senses) is None:
                break

            parts.insert(0, prev)
            start_i = j
            j -= 1

        span = "".join(parts)
        if cjk_count(span) < min_chars or len(parts) < 2:
            continue

        readings = []
        units = []
        valid = True
        for idx, part in enumerate(parts):
            r = morpheme_reading(part, morphemes, matched_terms, selected_senses)
            if not r:
                valid = False
                break
            readings.append(r)
            units.append({
                "source": part,
                "reading": r,
                "role": "HEAD" if idx == len(parts) - 1 else "MODIFIER",
            })
        if not valid:
            continue

        candidate = " ".join(readings)
        exact_term = _exact_matched_entry_for_span(span, matched_terms)
        semantic_authority = _compound_authority(span, cls, parts, compound_policy, morphemes)
        target_text_authority = bool((exact_term or {}).get("targetTextAuthority", False))
        out.append({
            "sourceSpan": span,
            "tokens": list(parts),
            "semanticUnits": units,
            "candidate": candidate,
            "class": cls,
            "anchor": anchor,
            "headReading": readings[-1],
            "modifierSpan": "".join(parts[:-1]),
            "startTokenIndex": start_i,
            "endTokenIndex": i,
            # Backward-compatible alias plus the two-axis FIX6.4.8 authority split.
            "authority": semantic_authority,
            "semanticAuthority": semantic_authority,
            "targetTextAuthority": target_text_authority,
            "knowledgeHardness": str((exact_term or {}).get("hardness", "soft")).upper(),
            "sourceMutation": False,
        })

    # Deduplicate structural duplicates.
    seen, dedup = set(), []
    for row in out:
        k = (row["startTokenIndex"], row["endTokenIndex"], row["sourceSpan"], row["candidate"])
        if k not in seen:
            seen.add(k)
            dedup.append(row)

    # Build a containment graph and mark maximal spans primary.
    for row in dedup:
        parents = []
        for other in dedup:
            if other is row:
                continue
            contains = (
                other["startTokenIndex"] <= row["startTokenIndex"]
                and other["endTokenIndex"] >= row["endTokenIndex"]
                and (
                    other["startTokenIndex"] < row["startTokenIndex"]
                    or other["endTokenIndex"] > row["endTokenIndex"]
                )
            )
            if contains:
                parents.append(other["sourceSpan"])
        row["containedBy"] = sorted(set(parents), key=len)
        row["isPrimary"] = not bool(parents)

    return dedup




class SessionEntityRegistry:
    """Ephemeral per-run entity memory. Never persisted and never mutates source."""
    def __init__(self):
        self._rows = {}

    def get(self, source_span):
        row = self._rows.get(source_span)
        return dict(row) if row else None

    def rows_for_source(self, source):
        out = []
        for span, row in self._rows.items():
            if span and span in source:
                r = dict(row)
                r["evidence"] = list(r.get("evidence") or []) + ["session-entity-registry"]
                out.append(r)
        return out

    def remember(self, row):
        if not row.get("targetTextAuthority"):
            return
        span = row.get("sourceSpan", "")
        canonical = row.get("canonicalTarget", "")
        if not span or not canonical:
            return
        self._rows[span] = {
            "sourceSpan": span,
            "candidateType": row.get("candidateType"),
            "canonicalTarget": canonical,
            "semanticAuthority": row.get("semanticAuthority", "STRONG"),
            "targetTextAuthority": True,
            "evidence": list(row.get("evidence") or []) + ["session-verified-by-current-evidence"],
        }


def _entry_forms(entry):
    return [x for x in [entry.get("source", "")] + list(entry.get("aliases") or []) if x]


def _span_context(source, span, radius=12):
    pos = source.find(span)
    if pos < 0:
        return "", ""
    return source[max(0, pos-radius):pos], source[pos+len(span):pos+len(span)+radius]


def _named_cue_for_span(source, span, cues, max_distance=10):
    left, _ = _span_context(source, span, max_distance + 8)
    for cue in cues or []:
        pos = left.rfind(cue)
        if pos >= 0 and len(left) - (pos + len(cue)) <= max_distance:
            return cue
    return None


def _artifact_identity_after(source, span, cues, max_distance=14):
    _, right = _span_context(source, span, max_distance + 8)
    compact = right.replace(" ", "")
    return next((cue for cue in (cues or []) if cue and cue in compact[:max_distance+8]), None)


def _ordinary_object_evidence(source, span, comp, morphemes, policy):
    left, right = _span_context(source, span, 14)
    local = left + span + right
    cue = next((x for x in policy.get("ordinaryObjectCues", []) if x and x in local), None)
    if cue:
        return True, f"ordinary-cue:{cue}"

    if comp and comp.get("class") == "artifact":
        modifier_cats = []
        for unit in comp.get("semanticUnits", [])[:-1]:
            e = morphemes.get(unit.get("source", ""), {})
            if e.get("category"):
                modifier_cats.append(e.get("category"))
        ordinary = set(policy.get("ordinaryModifierCategories", []))
        domain = set(policy.get("domainModifierCategories", []))
        eligible_heads = set(policy.get("ordinaryCompositionEligibleHeads", []))
        if (comp.get("anchor") in eligible_heads and modifier_cats
                and all(c in ordinary for c in modifier_cats)
                and not any(c in domain for c in modifier_cats)):
            return True, "material/color-modifier-composition"
    return False, None


def _cue_driven_named_spans(source, matched_terms, morphemes, special_policy):
    """Extract whole named entities after explicit naming grammar before word-by-word logic."""
    rows = []
    max_chars = int(special_policy.get("maxCueNamedSpanChars", 10))
    groups = [
        ("NAMED_ARTIFACT", special_policy.get("namedArtifactCuesBefore", []), set(special_policy.get("namedArtifactHeads", []))),
        ("NAMED_TECHNIQUE", special_policy.get("namedTechniqueCuesBefore", []), set(special_policy.get("namedTechniqueHeads", []))),
        ("NAMED_DOCUMENT", special_policy.get("namedDocumentCuesBefore", []), set(special_policy.get("namedDocumentHeads", []))),
        ("NAMED_CREATURE", special_policy.get("namedCreatureCuesBefore", []), set(special_policy.get("namedCreatureHeads", []))),
    ]
    for ctype, cues, heads in groups:
        ordered_cues = sorted(set(cues or []), key=len, reverse=True)
        for cue in ordered_cues:
            start = 0
            while cue:
                pos = source.find(cue, start)
                if pos < 0:
                    break
                start = pos + len(cue)
                # If a longer naming cue begins at the same position, only the longest
                # cue owns this span. This prevents partial cues such as æ­¤å®å from
                # producing a bogus entity beginning with ä¸º in æ­¤å®åä¸ºX.
                if any(len(longer) > len(cue) and source.startswith(longer, pos) for longer in ordered_cues):
                    continue
                tail = source[start:start + max_chars]
                cjk = ""
                for ch in tail:
                    if not is_cjk(ch):
                        break
                    cjk += ch
                if not cjk:
                    continue
                endpoints = []
                for i in range(2, len(cjk) + 1):
                    pref = cjk[:i]
                    if any(pref.endswith(h) for h in heads if h):
                        endpoints.append(pref)
                if not endpoints:
                    continue
                span = max(endpoints, key=len)
                reading = morpheme_reading(span, morphemes, matched_terms, [])
                if not reading:
                    # Whole-span identity is still useful, but without a trustworthy
                    # Vietnamese canonical surface it must not become a text lock.
                    rows.append({
                        "sourceSpan": span, "candidateType": ctype, "semanticClass": "cue-named-entity",
                        "canonicalTarget": "", "targetCandidates": [], "semanticAuthority": "STRONG",
                        "targetTextAuthority": False, "criticalForQa": True,
                        "renderingMode": "ENTITY_IDENTITY_REVIEW", "evidence": [f"cue-driven:{cue}", "canonical-unresolved"],
                        "sourceMutation": False,
                    })
                else:
                    rows.append({
                        "sourceSpan": span, "candidateType": ctype, "semanticClass": "cue-named-entity",
                        "canonicalTarget": reading, "targetCandidates": [reading], "semanticAuthority": "STRONG",
                        "targetTextAuthority": bool(special_policy.get("namedEntityTargetTextAuthority", True)),
                        "criticalForQa": True, "renderingMode": "CANONICAL_LOCK",
                        "evidence": [f"cue-driven:{cue}", "whole-span-before-word-by-word"], "sourceMutation": False,
                    })
    return rows


def _special_row_rank(row, priority):
    return (
        1 if row.get("targetTextAuthority") else 0,
        int(priority.get(row.get("candidateType", "UNCERTAIN"), 0)),
        1 if str(row.get("semanticAuthority", "SOFT")).upper() == "STRONG" else 0,
        len(row.get("sourceSpan", "")),
    )


def _suppress_contained_special_rows(rows, source, priority):
    # FIX6.4.8: a cue-driven/verified whole named entity owns all contained child spans.
    # This prevents a fragment such as ç•Œæ—— from overriding çŽ„é£Žå®šç•Œæ——.
    named_parent_types = {"NAMED_ARTIFACT", "NAMED_TECHNIQUE", "NAMED_DOCUMENT", "NAMED_CREATURE", "NAMED_PLACE_OR_SECT"}
    out = []
    for row in rows:
        span = row.get("sourceSpan", "")
        if not span:
            continue
        rrank = _special_row_rank(row, priority)
        suppressed = False
        for parent in rows:
            pspan = parent.get("sourceSpan", "")
            if parent is row or not pspan or len(pspan) <= len(span):
                continue
            if span not in pspan:
                continue
            if pspan not in source:
                continue
            if parent.get("candidateType") in named_parent_types and (
                any(str(e).startswith("cue-driven:") for e in (parent.get("evidence") or []))
                or "session-entity-registry" in (parent.get("evidence") or [])
            ):
                suppressed = True
                break
            prank = _special_row_rank(parent, priority)
            # A locked child entity from session memory must survive a larger noisy
            # compositional span. Otherwise prefer the stronger/longer whole span.
            if row.get("targetTextAuthority") and not parent.get("targetTextAuthority"):
                continue
            if prank >= rrank:
                suppressed = True
                break
        if not suppressed:
            out.append(row)
    return out


def detect_special_spans(source, matched_terms, names, sino_compounds, morphemes, special_policy, entity_registry=None):
    """FIX6.4.8 WHOLE-SPAN BEFORE WORD-BY-WORD resolver.

    It classifies spans and separates semanticAuthority from targetTextAuthority.
    It never rewrites SOURCE and contains no sentence-specific target mapping.
    """
    rows = []
    priority = special_policy.get("typePriority", {})
    rows.extend(_cue_driven_named_spans(source, matched_terms, morphemes, special_policy))
    if entity_registry:
        current_person_surnames = {
            n.get("surnameSurface", "") for n in (names or [])
            if n.get("surnameEvidence") == "EXACT_COMMON_SURNAME"
        }
        for remembered in entity_registry.rows_for_source(source):
            if (remembered.get("candidateType") == "PERSON_NAME"
                    and remembered.get("sourceSpan", "") not in current_person_surnames):
                continue
            rows.append(remembered)
    cat_map = special_policy.get("categoryTypeMap", {})
    comp_map = special_policy.get("compoundClassTypeMap", {})
    max_dist = int(special_policy.get("maxNamedCueDistanceChars", 10))

    # Exact proper-name evidence: canonical surname rendering is text-authoritative
    # only after the span has actually been classified as a person name.
    for n in names or []:
        if n.get("surnameEvidence") != "EXACT_COMMON_SURNAME":
            continue
        span = n.get("surnameSurface", "")
        canonical = n.get("surnameVietnamese", "")
        if span and canonical:
            rows.append({
                "sourceSpan": span,
                "candidateType": "PERSON_NAME",
                "semanticClass": "person-name",
                "canonicalTarget": canonical,
                "targetCandidates": [canonical],
                "semanticAuthority": "STRONG",
                "targetTextAuthority": bool(special_policy.get("nameTargetTextAuthority", True)),
                "criticalForQa": True,
                "renderingMode": "CANONICAL_LOCK",
                "evidence": ["exact-common-surname", f"title:{n.get('titleSource','')}"] if n.get("titleSource") else ["exact-common-surname"],
                "sourceMutation": False,
            })

    # Curated whole-span lexical knowledge.
    for e in matched_terms or []:
        surface = next((f for f in _entry_forms(e) if f in source), None)
        if not surface:
            continue
        cat = e.get("category", "")
        ctype = cat_map.get(cat, "GENERIC_DOMAIN_TERM")
        hardness = str(e.get("hardness", "soft")).upper()
        semantic_authority = "STRONG" if hardness == "STRONG" or e.get("criticalForQa") else "SOFT"
        text_authority = bool(e.get("targetTextAuthority", False))
        evidence = [f"knowledge:{e.get('_pack','')}", f"category:{cat}"]

        if cat == "idiom":
            ctype = "IDIOM_FIXED_EXPRESSION"
            semantic_authority = "STRONG" if e.get("criticalForQa") or float(e.get("confidence", 0)) >= 0.95 else semantic_authority
            text_authority = False
            rendering_mode = "WHOLE_SPAN_MEANING"
        else:
            rendering_mode = "CANONICAL_LOCK" if text_authority else "SEMANTIC_EQUIVALENT"

        # Naming grammar can promote an artifact/technique to a proper entity without
        # turning every ordinary object of that category into a fixed name.
        if cat in {"artifact", "weapon", "talisman", "formation_device"}:
            cue = _named_cue_for_span(source, surface, special_policy.get("namedArtifactCuesBefore", []), max_dist)
            after = _artifact_identity_after(source, surface, special_policy.get("artifactIdentityCuesAfter", []))
            if cue or after:
                ctype = "NAMED_ARTIFACT"
                text_authority = bool(special_policy.get("namedEntityTargetTextAuthority", True))
                semantic_authority = "STRONG"
                rendering_mode = "CANONICAL_LOCK"
                evidence.append(f"named-cue:{cue or after}")
        elif cat in {"technique", "formation_document", "document"}:
            cue = _named_cue_for_span(source, surface, special_policy.get("namedTechniqueCuesBefore", []) + special_policy.get("namedDocumentCuesBefore", []), max_dist)
            if cue:
                ctype = "NAMED_DOCUMENT" if cat in {"formation_document", "document"} else "NAMED_TECHNIQUE"
                text_authority = bool(special_policy.get("namedEntityTargetTextAuthority", True))
                semantic_authority = "STRONG"
                rendering_mode = "CANONICAL_LOCK"
                evidence.append(f"named-cue:{cue}")

        candidates = list(e.get("targetCandidates") or [])
        preferred = e.get("preferredTarget", "")
        if preferred and preferred not in candidates:
            candidates.insert(0, preferred)
        rows.append({
            "sourceSpan": surface,
            "candidateType": ctype,
            "semanticClass": cat,
            "canonicalTarget": preferred,
            "targetCandidates": candidates,
            "semanticAuthority": semantic_authority,
            "targetTextAuthority": text_authority,
            "criticalForQa": bool(e.get("criticalForQa", False)),
            "renderingMode": rendering_mode,
            "evidence": evidence,
            "sourceMutation": False,
        })

    # Compositional spans: namedness is evidence-driven, not length-driven.
    for comp in primary_sino_compounds(sino_compounds or []):
        span = comp.get("sourceSpan", "")
        if not span:
            continue
        cls = comp.get("class", "")
        ctype = comp_map.get(cls, "UNCERTAIN")
        semantic_authority = str(comp.get("semanticAuthority", comp.get("authority", "SOFT"))).upper()
        text_authority = bool(comp.get("targetTextAuthority", False))
        evidence = [f"compound-class:{cls}", f"semantic-authority:{semantic_authority}"]

        ordinary, ordinary_evidence = _ordinary_object_evidence(source, span, comp, morphemes, special_policy)
        if ordinary:
            ctype = "ORDINARY_OBJECT"
            text_authority = False
            evidence.append(ordinary_evidence)

        cue = None
        if cls in {"artifact", "formation_talisman"} and not ordinary:
            cue = _named_cue_for_span(source, span, special_policy.get("namedArtifactCuesBefore", []), max_dist)
            after = _artifact_identity_after(source, span, special_policy.get("artifactIdentityCuesAfter", []))
            if cue or after:
                ctype = "NAMED_ARTIFACT"
                semantic_authority = "STRONG"
                text_authority = bool(special_policy.get("namedEntityTargetTextAuthority", True))
                evidence.append(f"named-cue:{cue or after}")
        elif cls == "technique_text":
            cue = _named_cue_for_span(source, span, special_policy.get("namedTechniqueCuesBefore", []) + special_policy.get("namedDocumentCuesBefore", []), max_dist)
            if cue:
                is_doc = comp.get("anchor") in set(special_policy.get("namedDocumentHeads", []))
                ctype = "NAMED_DOCUMENT" if is_doc else "NAMED_TECHNIQUE"
                semantic_authority = "STRONG"
                text_authority = bool(special_policy.get("namedEntityTargetTextAuthority", True))
                evidence.append(f"named-cue:{cue}")

        remembered = entity_registry.get(span) if entity_registry else None
        canonical = comp.get("candidate", "")
        if remembered:
            ctype = remembered.get("candidateType") or ctype
            canonical = remembered.get("canonicalTarget") or canonical
            semantic_authority = "STRONG"
            text_authority = True
            evidence.append("session-entity-registry")

        rendering_mode = "CANONICAL_LOCK" if text_authority else "SEMANTIC_EQUIVALENT"
        rows.append({
            "sourceSpan": span,
            "candidateType": ctype,
            "semanticClass": cls,
            "canonicalTarget": canonical,
            "targetCandidates": [canonical] if canonical else [],
            "semanticAuthority": semantic_authority,
            "targetTextAuthority": text_authority,
            "criticalForQa": bool(text_authority),
            "renderingMode": rendering_mode,
            "evidence": evidence,
            "sourceMutation": False,
        })

    # Deduplicate same source span, then suppress weaker contained spans so guidance/QA
    # binds to the maximal semantic entity rather than nested fragments.
    best = {}
    for row in rows:
        span = row.get("sourceSpan", "")
        rank = _special_row_rank(row, priority)
        if span not in best or rank > best[span][0]:
            best[span] = (rank, row)
    out = _suppress_contained_special_rows([x[1] for x in best.values()], source, priority)
    out.sort(key=lambda r: (source.find(r.get("sourceSpan", "")), -len(r.get("sourceSpan", ""))))
    if entity_registry:
        for row in out:
            entity_registry.remember(row)
    return out




def _creature_named_parent(span, special_spans):
    for row in special_spans or []:
        p=row.get("sourceSpan","")
        if p and span in p and row.get("candidateType")=="NAMED_CREATURE":
            return row
    return None


def detect_creature_semantics(source, special_spans, creature_policy):
    """Generic creature/ecology composition with body/trace semantics.

    Preserve class/species identity and salient modifiers. Natural Vietnamese is allowed;
    this never creates a sentence answer map and never rewrites SOURCE/TARGET.
    """
    rows=[]; occupied=[]
    # Exact creature classes first.
    for e in sorted(creature_policy.get("classEntries",[]), key=lambda x:len(x.get("source","")), reverse=True):
        span=e.get("source","")
        start=0
        while span:
            pos=source.find(span,start)
            if pos<0: break
            start=pos+len(span)
            if _creature_named_parent(span,special_spans): continue
            rows.append({"sourceSpan":span,"creatureType":"CREATURE_CLASS","semanticClass":"creature-class","headSource":span,"headCandidates":list(e.get("targetCandidates") or []),"authority":e.get("authority","STRONG"),"components":[],"targetTextAuthority":False,"sourceMutation":False})
            occupied.append((pos,pos+len(span)))

    # Species phrase = modifiers + species head.
    heads=sorted(creature_policy.get("speciesHeads",[]), key=lambda x:len(x.get("source","")), reverse=True)
    mods=sorted(creature_policy.get("modifierEntries",[]), key=lambda x:len(x.get("source","")), reverse=True)
    max_left=int(creature_policy.get("maxLeftModifierChars",6))
    for h in heads:
        hs=h.get("source","")
        start=0
        while hs:
            pos=source.find(hs,start)
            if pos<0: break
            start=pos+len(hs)
            if any(a<=pos and pos+len(hs)<=b for a,b in occupied): continue
            left=pos; comps=[]
            while left>0 and pos-left < max_left:
                m=next((m for m in mods if m.get("source") and left>=len(m["source"]) and source[left-len(m["source"]):left]==m["source"]),None)
                if not m: break
                ms=m['source']; left-=len(ms)
                comps.insert(0,{"source":ms,"role":m.get("role","ATTRIBUTE"),"targetCandidates":list(m.get("targetCandidates") or [])})
            span=source[left:pos+len(hs)]
            if len(span)==1 and hs=='å…½': continue
            if _creature_named_parent(span,special_spans): continue
            rows.append({"sourceSpan":span,"creatureType":"SPECIES_PHRASE","semanticClass":"creature-species","headSource":hs,"headCandidates":list(h.get("targetCandidates") or []),"authority":h.get("authority","STRONG"),"components":comps,"targetTextAuthority":False,"sourceMutation":False})

    # Creature body parts/traces: species/class immediately before a body head is semantic,
    # e.g. è›Ÿé³ž = váº£y giao, å…½çˆªå° = dáº¥u vuá»‘t thÃº. This is compositional, not namedness.
    body_heads=sorted(creature_policy.get("bodyPartHeads",[]), key=lambda x:len(x.get("source","")), reverse=True)
    species_for_body=sorted(
        [{"source":x.get("source"),"targetCandidates":x.get("targetCandidates",[])} for x in heads]
        + [{"source":"å…½","targetCandidates":["thÃº"]}],
        key=lambda x:len(x.get("source","") or ""), reverse=True
    )
    for bh in body_heads:
        hs=bh.get("source","")
        start=0
        while hs:
            pos=source.find(hs,start)
            if pos<0: break
            start=pos+len(hs)
            left=pos; comps=[]
            sm=next((x for x in species_for_body if x.get("source") and left>=len(x["source"]) and source[left-len(x["source"]):left]==x["source"]),None)
            if sm:
                left-=len(sm['source'])
                comps.append({"source":sm['source'],"role":"SPECIES_MODIFIER","targetCandidates":list(sm.get('targetCandidates') or [])})
            span=source[left:pos+len(hs)]
            if _creature_named_parent(span,special_spans): continue
            rows.append({"sourceSpan":span,"creatureType":"CREATURE_BODY_OR_TRACE","semanticClass":"creature-body-trace","headSource":hs,"headCandidates":list(bh.get("targetCandidates") or []),"authority":bh.get("authority","STRONG"),"components":comps,"targetTextAuthority":False,"sourceMutation":False})

    # Keep maximal overlapping rows within the same semantic family; distinct class/body rows survive.
    out=[]
    for row in rows:
        span=row.get('sourceSpan',''); spos=source.find(span); send=spos+len(span)
        if row.get('creatureType') in {'SPECIES_PHRASE','CREATURE_BODY_OR_TRACE'} and any(
            o is not row and o.get('creatureType')==row.get('creatureType') and len(o.get('sourceSpan',''))>len(span)
            and source.find(o['sourceSpan'])<=spos and source.find(o['sourceSpan'])+len(o['sourceSpan'])>=send
            for o in rows
        ):
            continue
        key=(row.get('sourceSpan'),row.get('creatureType'))
        if not any((x.get('sourceSpan'),x.get('creatureType'))==key for x in out): out.append(row)
    out.sort(key=lambda r:(source.find(r.get('sourceSpan','')),-len(r.get('sourceSpan',''))))
    return out

def creature_semantic_check(target, creature_rows, creature_policy):
    warnings=[]; critical=[]
    critical_roles=set(creature_policy.get('criticalRoles',[]))
    for row in creature_rows or []:
        heads=row.get('headCandidates') or []
        if heads and not contains_any_candidate(target,heads):
            critical.append(f"CREATURE_HEAD_LOSS {row.get('sourceSpan')} head={row.get('headSource')} expected={heads}")
        for c in row.get('components') or []:
            cands=c.get('targetCandidates') or []
            if cands and not contains_any_candidate(target,cands):
                msg=f"CREATURE_COMPONENT_LOSS {row.get('sourceSpan')} component={c.get('source')} role={c.get('role')} expected={cands}"
                (critical if c.get('role') in critical_roles else warnings).append(msg)
    return warnings,critical



def _token_span_rows(source, tokens):
    """Return token rows with source offsets; used only for internal semantic framing."""
    return [
        {"source": tok, "start": st, "end": en}
        for tok, st, en in _token_offsets(source, tokens)
    ]


def _frame_slot(source, start, end, role, *, confidence="MEDIUM", candidates=None, authority="SOFT", evidence=None):
    return {
        "sourceSpan": source[start:end],
        "sourceStart": start,
        "sourceEnd": end,
        "role": role,
        "confidence": confidence,
        "targetCandidates": list(candidates or []),
        "semanticAuthority": str(authority or "SOFT").upper(),
        "evidence": list(evidence or []),
    }


def _strong_target_candidates(row):
    if not row:
        return []
    cands = list(row.get("targetCandidates") or [])
    pref = row.get("preferredTarget") or row.get("canonicalTarget")
    if pref and pref not in cands:
        cands.insert(0, pref)
    authority = str(row.get("semanticAuthority", row.get("qaAuthority", row.get("authority", "SOFT")))).upper()
    text_authority = bool(row.get("targetTextAuthority"))
    return cands if authority == "STRONG" or text_authority else []


def build_meaning_frame(source, tokens, analysis, special_spans=None, quantity_frames=None,
                        noun_phrases=None, creature_rows=None, selected_senses=None,
                        matched_terms=None, structural_senses=None):
    """FIX6.4.9-A internal semantic representation.

    This is deliberately conservative: it records evidence already produced by the
    existing analyzers instead of inventing a second parser.  It never mutates SOURCE,
    never emits an expected translation, and is not sent wholesale to HY-MT.
    """
    token_rows = _token_span_rows(source, tokens)
    frame = {
        "version": "FIX6.4.9-A",
        "subject": [], "predicate": [], "object": [], "negation": [], "modality": [],
        "quantity": [], "duration": [], "entity": [], "domain": [], "modifier": [],
        "relation": [], "body_part": [], "new_feature": [],
    }

    # High-confidence lexical roles from the existing grammar lexicon.
    verb_set = set(load_grammar().get("common_verbs", [])) if False else None
    # Avoid re-loading KB in the hot path: common verbs are carried by token classes below.
    # Predicate is inferred only when the grammar category itself says it is a common verb.
    grammar_verbs = set(_MEANING_FRAME_COMMON_VERBS)
    # Chinese predicates may be segmented into adjacent characters (e.g. é•¿ + å‡º).
    # Resolve the longest grammar-known verb surface directly against SOURCE.
    predicate_candidates = []
    for verb in grammar_verbs:
        pos = source.find(verb)
        if pos >= 0:
            predicate_candidates.append((len(verb), pos, verb))
    predicate_row = None
    if predicate_candidates:
        _, pstart, pverb = max(predicate_candidates, key=lambda x: (x[0], -x[1]))
        predicate_row = {"source": pverb, "start": pstart, "end": pstart + len(pverb)}
        frame["predicate"].append(_frame_slot(
            source, pstart, pstart + len(pverb), "PREDICATE",
            confidence="MEDIUM", evidence=["grammar:common_verb_surface"]
        ))

    # High-confidence subject evidence comes from the maximal creature species/class span
    # that occurs before the predicate. Nested creature features after the predicate are not subjects.
    predicate_start = predicate_row["start"] if predicate_row else len(source)
    subject_rows = []
    for row in creature_rows or []:
        if row.get("creatureType") not in {"CREATURE_CLASS", "SPECIES_PHRASE"} or not row.get("sourceSpan"):
            continue
        pos = source.find(row.get("sourceSpan", ""))
        if 0 <= pos < predicate_start:
            subject_rows.append((pos, len(row.get("sourceSpan", "")), row))
    if subject_rows:
        _, _, row = max(subject_rows, key=lambda x: (x[0], x[1]))
        span = row.get("sourceSpan")
        pos = source.find(span)
        if pos > 0 and not is_cjk(source[pos-1]) is False:
            # Preserve an immediately attached lexical modifier such as å·¨ in å·¨èŸ’.
            if is_cjk(source[pos-1]) and source[pos-1] not in _MEANING_FRAME_FUNCTIONAL:
                span = source[pos-1:pos+len(row.get("sourceSpan"))]
        frame["subject"].append({
            "sourceSpan": span, "role": "SUBJECT", "confidence": "HIGH",
            "targetCandidates": _strong_target_candidates(row),
            "semanticAuthority": str(row.get("authority", "STRONG")).upper(),
            "evidence": ["maximal-creature-species/class-before-predicate"]
        })

    # Strong noun-phrase heads are safe object evidence; SOFT ordinary objects remain free.
    for row in noun_phrases or []:
        cands = _strong_target_candidates(row)
        if cands:
            frame["object"].append({
                "sourceSpan": row.get("sourceSpan", ""), "role": "OBJECT", "confidence": "HIGH",
                "targetCandidates": cands,
                "semanticAuthority": str(row.get("authority", "SOFT")).upper(),
                "evidence": ["noun-phrase-head"]
            })


    for occ in (analysis or {}).get("negationOccurrences", []):
        frame["negation"].append(_frame_slot(
            source, occ["start"], occ["end"], "NEGATION", confidence="HIGH",
            evidence=["grammar:negation"]
        ))
    for tok, st, en in _token_offsets(source, tokens):
        if tok in set((analysis or {}).get("modality", [])):
            frame["modality"].append(_frame_slot(
                source, st, en, "MODALITY", confidence="HIGH", evidence=["grammar:modality"]
            ))

    for q in quantity_frames or []:
        role = q.get("role")
        key = "duration" if str(role).startswith("DURATION") else "quantity"
        cands = list(q.get("targetCandidates") or [])
        frame[key].append({
            "sourceSpan": q.get("sourceSpan", ""), "sourceStart": q.get("sourceStart"),
            "sourceEnd": q.get("sourceEnd"), "role": role, "confidence": "HIGH",
            "targetCandidates": cands, "semanticAuthority": str(q.get("authority", "SOFT")).upper(),
            "evidence": ["quantity-frame"]
        })

    for row in special_spans or []:
        slot = "entity" if row.get("candidateType") != "IDIOM_FIXED_EXPRESSION" else "relation"
        cands = _strong_target_candidates(row)
        frame[slot].append({
            "sourceSpan": row.get("sourceSpan", ""), "sourceStart": row.get("sourceStart"),
            "sourceEnd": row.get("sourceEnd"), "role": row.get("candidateType"),
            "confidence": "HIGH", "targetCandidates": cands,
            "semanticAuthority": str(row.get("semanticAuthority", "SOFT")).upper(),
            "evidence": ["special-span"]
        })

    for row in creature_rows or []:
        cspan = row.get("sourceSpan", "")
        cpos = source.find(cspan) if cspan else -1
        creature_evidence = len(cspan) > 1
        if len(cspan) == 1 and cpos > 0:
            # Single-character creature body heads require an explicit creature lexical
            # modifier/class in the immediately preceding source span.  A generic noun
            # such as æ¡Œè§’ therefore cannot acquire CREATURE ownership.
            preceding = source[max(0, cpos-3):cpos]
            creature_evidence = any(
                r.get("creatureType") in {"CREATURE_CLASS", "SPECIES_PHRASE"}
                and r.get("sourceSpan")
                and preceding.endswith(r.get("sourceSpan"))
                for r in (creature_rows or [])
                if r is not row
            )
        if not creature_evidence:
            continue
        frame["domain"].append({
            "sourceSpan": row.get("sourceSpan", ""), "role": "CREATURE",
            "confidence": "HIGH", "targetCandidates": _strong_target_candidates(row),
            "semanticAuthority": str(row.get("authority", "STRONG")).upper(),
            "evidence": ["creature-semantic-composer"]
        })
        hs = row.get("headSource")
        if hs and row.get("creatureType") == "CREATURE_BODY_OR_TRACE":
            frame["body_part"].append({
                "sourceSpan": row.get("sourceSpan", ""), "role": "BODY_OR_TRACE",
                "confidence": "HIGH", "targetCandidates": _strong_target_candidates(row),
                "semanticAuthority": str(row.get("authority", "STRONG")).upper(),
                "evidence": ["creature-body-trace"]
            })
            # A preceding lexical body-location head (e.g. è…¹éƒ¨) is a relation, not the
            # creature feature itself.  The generic éƒ¨ suffix avoids sentence-specific lists.
            pos = source.find(row.get("sourceSpan", ""))
            if pos > 0:
                left = source[max(0, pos-8):pos]
                matches = list(re.finditer(r"([^\W_])éƒ¨", left))
                if matches:
                    m = matches[-1]
                    loc = m.group(0)
                    loc_start = pos - len(left) + m.start()
                    frame["body_part"].insert(0, {
                        "sourceSpan": loc, "sourceStart": loc_start, "sourceEnd": pos,
                        "role": "BODY_LOCATION", "confidence": "HIGH",
                        "targetCandidates": [], "semanticAuthority": "SOFT",
                        "evidence": ["lexical-body-location-suffix"]
                    })

    for row in (matched_terms or []) + (selected_senses or []) + (structural_senses or []):
        cands = _strong_target_candidates(row)
        if not cands:
            continue
        role = row.get("semanticRole") or "DOMAIN_TERM"
        slot = "domain" if role in {"FORMATION", "CREATURE", "ITEM", "ARTIFACT", "ALCHEMY", "TECHNIQUE", "DOMAIN"} else "modifier"
        frame[slot].append({
            "sourceSpan": row.get("source") or row.get("surface") or "", "role": role,
            "confidence": "HIGH", "targetCandidates": cands,
            "semanticAuthority": str(row.get("semanticAuthority", row.get("qaAuthority", "STRONG"))).upper(),
            "evidence": ["knowledge-or-contextual-sense"]
        })

    # De-duplicate identical evidence rows while preserving insertion order.
    for key, rows in frame.items():
        if not isinstance(rows, list):
            continue
        seen = set(); clean = []
        for row in rows:
            marker = (row.get("sourceSpan"), row.get("role"), tuple(row.get("targetCandidates") or []))
            if marker in seen:
                continue
            seen.add(marker); clean.append(row)
        frame[key] = clean
    return frame


def meaning_frame_coverage_check(target, frame):
    """Check only high-confidence, authoritative slots; ordinary semantic equivalents remain free."""
    warnings, critical = [], []
    for slot, rows in (frame or {}).items():
        for row in rows or []:
            cands = row.get("targetCandidates") or []
            if not cands or row.get("confidence") != "HIGH":
                continue
            authority = str(row.get("semanticAuthority", "SOFT")).upper()
            if authority != "STRONG":
                continue
            if not contains_any_candidate(target, cands):
                msg = f"MEANING_FRAME_COVERAGE_LOSS slot={slot} span={row.get('sourceSpan')} role={row.get('role')} expected={cands}"
                critical.append(msg)
    return warnings, critical



# Compact lexical-role vocabulary used by MeaningFrame; loaded once, never benchmark-specific.
_MEANING_FRAME_COMMON_VERBS = {
    "æ˜¯","æœ‰","å‡ºçŽ°","æ‰“å¼€","å…³é—­","éœ€è¦","å¸®åŠ©","çœ‹ç€","çœ‹è§","çœ‹åˆ°","é‡åˆ°","å‘ç”Ÿ","å‡ºäº‹","ç‰ºç‰²",
    "äº¤ç»™","äº¤ä»˜","ç•™ä¸‹","ä½™ä¸‹","å‰©ä¸‹","å›žæ¥","è¿›å…¥","ç¦»å¼€","å¾—åˆ°","æ‹¿åˆ°","å–å‡º","æ”¾ä¸‹","å‘çŽ°",
    "é•¿å‡º","åŽ‹ç€","è¿›å…¥","å½¢æˆ","å¸ƒä¸‹","å¸ƒç½®","ç»“æˆ","ç ´","æ¯","å´©","ç¨³","é•‡","å€Ÿ","é—®","æ„¿æ„",
}
_MEANING_FRAME_FUNCTIONAL = {
    "æˆ‘","æˆ‘ä»¬","ä½ ","ä½ ä»¬","æ‚¨","ä»–","å¥¹","å®ƒ","ä»–ä»¬","å¥¹ä»¬","å®ƒä»¬","è¿™","é‚£","å…¶",
    "ä¸","æ²¡","æ²¡æœ‰","æœª","æ— ","éž","èŽ«","å‹¿","åˆ«","ä¸èƒ½","ä¸å¯","è¦","éœ€è¦","å¿…é¡»","åº”è¯¥",
    "å¯ä»¥","èƒ½å¤Ÿ","èƒ½","æ„¿","æƒ³","ä¹Ÿ","åˆ","è¿˜","å´","ä½†","ä½†æ˜¯","ä¸è¿‡","ç„¶è€Œ","è€Œ","è€Œä¸”",
    "å¹¶ä¸”","å› ä¸º","æ‰€ä»¥","è‹¥","å¦‚æžœ","è™½ç„¶","å³ä½¿","æ—¢ç„¶","é‚£ä¹ˆ","çš„","åœ°","å¾—","ä¹‹","æ‰€","è€…",
    "æŠŠ","è¢«","ç»™","å‘","å¯¹","äºŽ","ä»¥","ä¸º","ä»Ž","åœ¨","ä¸Ž","å’Œ","äº†","ç€","è¿‡"
}

def _span_contains(container, child):
    return bool(container and child and child in container)


def suppress_lower_semantic_spans(sino_compounds, noun_phrases, creature_rows, quantity_frames, special_spans):
    """Deterministic ownership precedence:
    named entity > quantity measure > creature phrase > noun phrase > HÃ¡n-Viá»‡t compound.
    It only suppresses conflicting analysis metadata; SOURCE/TARGET are never mutated.
    """
    named=[x.get('sourceSpan','') for x in (special_spans or []) if x.get('targetTextAuthority') and x.get('sourceSpan')]
    quantities=[x.get('sourceSpan','') for x in (quantity_frames or []) if x.get('role') in {'DURATION_MEASURE','DURATION_UNIT'} and x.get('sourceSpan')]
    creatures=[x.get('sourceSpan','') for x in (creature_rows or []) if x.get('sourceSpan')]
    nps=[x.get('sourceSpan','') for x in (noun_phrases or []) if x.get('sourceSpan')]
    owners=named+quantities+creatures+nps
    comps=[]
    for c in sino_compounds or []:
        cs=c.get('sourceSpan','')
        if cs and any(_span_contains(o,cs) for o in owners):
            continue
        comps.append(c)
    # quantity/creature ownership also removes spurious ordinary NP subheads (e.g. é’Ÿ inside åŠåˆ»é’Ÿ)
    np_out=[]
    for n in noun_phrases or []:
        ns=n.get('sourceSpan','')
        if ns and any(_span_contains(o,ns) for o in named+quantities+creatures):
            continue
        np_out.append(n)
    return comps,np_out


def _first_clean_candidate(candidates):
    for c in candidates or []:
        if c and cjk_count(str(c)) == 0:
            return str(c)
    return str((candidates or [""])[0]) if candidates else ""


def build_semantic_plan(matched_terms, special_spans, quantity_frames, noun_phrases, creature_rows, selected_senses):
    """Typed, compressed semantic plan for FIX6.4.8.

    This is NOT a target sentence. It selects at most three high-value lexical/semantic
    anchors, each derived from generic knowledge or current-source structure. Ordinary
    terms remain semantic-equivalent; only verified/named spans use exact text authority.
    """
    rows=[]
    def add(priority,kind,span,candidates,exact=False,components=None,critical=True):
        cands=[x for x in (candidates or []) if x]
        if not span or not cands: return
        rows.append({"priority":priority,"kind":kind,"sourceSpan":span,"targetCandidates":cands,
                     "exact":bool(exact),"components":components or [],"critical":bool(critical)})

    for sp in special_spans or []:
        cands=list(sp.get('targetCandidates') or [])
        can=sp.get('canonicalTarget')
        if can and can not in cands: cands.insert(0,can)
        if sp.get('targetTextAuthority') and can:
            add(100,'NAMED',sp.get('sourceSpan'),[can],True,critical=True)
        elif sp.get('candidateType')=='IDIOM_FIXED_EXPRESSION' and cands:
            add(95,'IDIOM',sp.get('sourceSpan'),cands,False,critical=bool(sp.get('criticalForQa',True)))

    for q in quantity_frames or []:
        if q.get('role')=='DURATION_MEASURE' and q.get('targetCandidates'):
            add(99,'DURATION',q.get('sourceSpan'),q.get('targetCandidates'),False,critical=True)
        elif q.get('role')=='CARDINAL_QUANTITY' and q.get('targetCandidates'):
            add(88,'QUANTITY',q.get('sourceSpan'),q.get('targetCandidates'),False,critical=True)

    for n in noun_phrases or []:
        if str(n.get('authority','SOFT')).upper()!='STRONG': continue
        comps=[c for c in (n.get('components') or []) if c.get('targetCandidates') and c.get('role') in {'MATERIAL','COLOR','DOMAIN_FUNCTION','STATE','DOMAIN_MODIFIER'}]
        add(96,'NP_HEAD',n.get('sourceSpan'),n.get('headCandidates'),False,components=comps,critical=True)

    for c in creature_rows or []:
        comps=[x for x in (c.get('components') or []) if x.get('targetCandidates')]
        add(97,'CREATURE',c.get('sourceSpan'),c.get('headCandidates'),False,components=comps,critical=True)

    for e in matched_terms or []:
        cands=list(e.get('targetCandidates') or [])
        pref=e.get('preferredTarget')
        if pref and pref not in cands: cands.insert(0,pref)
        hard=str(e.get('hardness','soft')).upper()=='STRONG' or bool(e.get('criticalForQa'))
        if hard and cands:
            add(93,'TERM',e.get('source'),cands,bool(e.get('targetTextAuthority')),critical=bool(e.get('criticalForQa',True)))

    for sense in selected_senses or []:
        if str(sense.get('qaAuthority','SOFT')).upper()!='STRONG': continue
        cands=list(sense.get('targetCandidates') or [])
        pref=sense.get('preferredTarget')
        if pref and pref not in cands: cands.insert(0,pref)
        add(92,'SENSE',sense.get('surface') or sense.get('source'),cands,False,critical=True)

    # Prefer the strongest semantic owner for a span; do not duplicate the same span.
    out=[]; seen=set()
    for row in sorted(rows,key=lambda x:(-x['priority'],-len(x['sourceSpan']))):
        span=row['sourceSpan']
        if span in seen: continue
        # A longer already-selected named/NP/creature span owns contained fragments.
        if any(span in old['sourceSpan'] and len(old['sourceSpan'])>len(span) for old in out):
            continue
        seen.add(span); out.append(row)
        if len(out)>=3: break
    return out


def render_semantic_plan(plan):
    parts=[]
    for row in plan or []:
        span=row.get('sourceSpan',''); main=_first_clean_candidate(row.get('targetCandidates'))
        if not span or not main: continue
        if row.get('kind')=='CREATURE' and row.get('components'):
            bits=[f"{row.get('sourceSpan')}={main}"]
            for c in row.get('components')[:2]:
                cv=_first_clean_candidate(c.get('targetCandidates'))
                if cv: bits.append(f"{c.get('source')}={cv}")
            parts.append(','.join(bits))
        elif row.get('kind')=='NP_HEAD' and row.get('components'):
            bits=[f"{row.get('sourceSpan')}={main}"]
            for c in row.get('components')[:1]:
                cv=_first_clean_candidate(c.get('targetCandidates'))
                if cv: bits.append(f"{c.get('source')}={cv}")
            parts.append(','.join(bits))
        else:
            parts.append(f"{span}={main}")
    return parts


def semantic_plan_coverage_check(target, plan):
    warnings=[]; critical=[]
    for row in plan or []:
        cands=row.get('targetCandidates') or []
        ok=contains_any_candidate(target,cands)
        if row.get('exact') and cands:
            ok=candidate_evidence(target,cands[0])
        if not ok:
            msg=f"SEMANTIC_PLAN_LOSS {row.get('sourceSpan')} kind={row.get('kind')} expected={cands}"
            (critical if row.get('critical',True) else warnings).append(msg)
        # For creature/NP components only enforce components that are intrinsically salient.
        for c in row.get('components') or []:
            role=c.get('role')
            if role not in {'MATERIAL','COLOR','DOMAIN_FUNCTION','DOMAIN_MODIFIER','SPECIES_MODIFIER','MORPHOLOGY','MORPHOLOGY_ELEMENT','ATTRIBUTE'}:
                continue
            cc=c.get('targetCandidates') or []
            if cc and not contains_any_candidate(target,cc):
                msg=f"SEMANTIC_PLAN_COMPONENT_LOSS {row.get('sourceSpan')} component={c.get('source')} role={role} expected={cc}"
                critical.append(msg)
    return warnings,critical

def _find_named_parent_for_span(span, special_spans, policy):
    named_types=set(policy.get("namedParentTypes",[]))
    for row in special_spans or []:
        p=row.get("sourceSpan","")
        if p and span in p and len(p)>=len(span) and row.get("candidateType") in named_types:
            return row
    return None


def detect_noun_phrase_semantics(source, special_spans, noun_policy):
    """Head-first noun-phrase composer for ordinary/domain objects.

    It does not produce a full Vietnamese sentence and does not make a named entity.
    A verified/cue-driven named parent always wins and suppresses ordinary NP rendering.
    """
    heads=sorted(noun_policy.get("headEntries",[]), key=lambda x:len(x.get("source","")), reverse=True)
    modifiers=sorted(noun_policy.get("modifierEntries",[]), key=lambda x:len(x.get("source","")), reverse=True)
    rows=[]; occupied=[]
    for h in heads:
        hs=h.get("source","")
        if not hs: continue
        start=0
        while True:
            pos=source.find(hs,start)
            if pos<0: break
            start=pos+len(hs)
            # Prefer longest head covering the same source region.
            if any(a<=pos and pos+len(hs)<=b for a,b in occupied):
                continue
            left_pos=pos; comps=[]; budget=int(noun_policy.get("maxLeftModifierChars",5))
            while left_pos>0 and pos-left_pos < budget:
                matched=None
                for mod in modifiers:
                    ms=mod.get("source","")
                    if ms and left_pos>=len(ms) and source[left_pos-len(ms):left_pos]==ms:
                        matched=mod; break
                if not matched: break
                ms=matched['source']; left_pos-=len(ms)
                comps.insert(0,{"source":ms,"role":matched.get("role","MODIFIER"),"targetCandidates":list(matched.get("targetCandidates") or [])})
            span=source[left_pos:pos+len(hs)]
            parent=_find_named_parent_for_span(span,special_spans,noun_policy)
            if parent:
                continue
            resolved_authority=h.get("authority","SOFT")
            if h.get("semanticClass")=="alchemy-ingredient" and any(c.get("role")=="DOMAIN_MODIFIER" for c in comps):
                resolved_authority="STRONG"
            row={
                "sourceSpan":span,"headSource":hs,"semanticClass":h.get("semanticClass","ordinary-object"),
                "headCandidates":list(h.get("targetCandidates") or []),"authority":resolved_authority,
                "components":comps,"renderingMode":"HEAD_FIRST_SEMANTIC_COMPOSITION",
                "targetTextAuthority":False,"sourceMutation":False,
            }
            rows.append(row); occupied.append((left_pos,pos+len(hs)))
    # Keep maximal overlapping NPs only.
    out=[]
    for row in rows:
        span=row['sourceSpan']; spos=source.find(span); send=spos+len(span)
        if any(source.find(o['sourceSpan'])<=spos and source.find(o['sourceSpan'])+len(o['sourceSpan'])>=send and len(o['sourceSpan'])>len(span) for o in rows):
            continue
        out.append(row)
    return out


def suppress_compounds_owned_by_noun_phrases(sino_compounds, noun_phrases):
    """NP semantic frames own their full source span over HÃ¡n-Viá»‡t compound guesses.

    This prevents fragments like é»‘é“ä»¤ or ç™½ç¬¦ from competing with the stronger
    head-first analyses é»‘é“ä»¤ç‰Œ and ç©ºç™½ç¬¦çº¸. Named entities are not noun phrases here.
    """
    spans=[x.get("sourceSpan","") for x in (noun_phrases or []) if x.get("sourceSpan")]
    out=[]
    for comp in sino_compounds or []:
        cs=comp.get("sourceSpan","")
        if cs and any(cs in ns for ns in spans):
            continue
        out.append(comp)
    return out


def noun_phrase_semantic_check(target, noun_phrases):
    warnings=[]; critical=[]
    for row in noun_phrases or []:
        heads=row.get('headCandidates') or []
        if heads and not contains_any_candidate(target,heads):
            msg=f"NP_HEAD_LOSS {row.get('sourceSpan')} head={row.get('headSource')} expected={heads}"
            (critical if str(row.get('authority','SOFT')).upper()=='STRONG' else warnings).append(msg)
        for c in row.get('components') or []:
            cands=c.get('targetCandidates') or []
            if cands and not contains_any_candidate(target,cands):
                msg=f"NP_COMPONENT_LOSS {row.get('sourceSpan')} component={c.get('source')} role={c.get('role')} expected={cands}"
                if c.get('role') in {'MATERIAL','COLOR','DOMAIN_FUNCTION','DOMAIN_MODIFIER'}:
                    critical.append(msg)
                else:
                    warnings.append(msg)
    return warnings,critical

def special_span_check(target, special_spans):
    warnings, critical = [], []
    for row in special_spans or []:
        candidates = list(row.get("targetCandidates") or [])
        canonical = row.get("canonicalTarget", "")
        if canonical and canonical not in candidates:
            candidates.insert(0, canonical)
        if not candidates:
            continue

        ctype = row.get("candidateType", "")
        if row.get("targetTextAuthority"):
            # Only proper/named/locked spans require canonical surface rendering.
            if canonical and not candidate_evidence(target, canonical):
                critical.append(
                    f"SPECIAL_SPAN_CANONICAL_LOSS {row.get('sourceSpan')} "
                    f"type={ctype} canonical={canonical}"
                )
            continue

        if ctype == "IDIOM_FIXED_EXPRESSION" and not contains_any_candidate(target, candidates):
            msg = f"WHOLE_SPAN_MEANING_VARIANT {row.get('sourceSpan')} expected={candidates}"
            (critical if row.get("criticalForQa") else warnings).append(msg)
    return warnings, critical

def compound_anchor_check(target, sino_compounds, compound_policy):
    warnings, critical = [], []
    nt = norm_vi(target)
    semantic_targets = compound_policy.get("semanticAnchorTargets", {})
    critical_anchors = set(compound_policy.get("criticalSemanticAnchors", []))

    for comp in primary_sino_compounds(sino_compounds):
        anchor = comp.get("anchor", "")
        allowed = semantic_targets.get(anchor, [])
        if not allowed:
            continue
        if any(norm_vi(x) in nt for x in allowed if x):
            continue

        msg = (
            f"COMPOUND_HEAD_LOSS {comp.get('sourceSpan')} anchor={anchor} "
            f"requires semantic head {allowed}"
        )
        if comp.get("authority") == "STRONG" or anchor in critical_anchors:
            critical.append(msg)
        else:
            warnings.append(msg)

    return warnings, critical


def compound_composition_check(target, sino_compounds, compound_policy):
    """
    Structural preservation check for high-confidence multi-token HÃ¡n-Viá»‡t compounds.
    It checks the whole composition and each semantic unit; it never rewrites target.
    """
    warnings, critical = [], []
    for comp in primary_sino_compounds(sino_compounds):
        if str(comp.get("semanticAuthority", comp.get("authority", "SOFT"))).upper() != "STRONG":
            continue
        # FIX6.4.8: semantic strength does not imply exact HÃ¡n-Viá»‡t surface authority.
        # Ordinary objects and soft domain terms are validated by semantic head/sense QA.
        if not bool(comp.get("targetTextAuthority", False)):
            continue

        candidate = comp.get("candidate", "")
        if candidate and candidate_evidence(target, candidate):
            continue

        missing = []
        for unit in comp.get("semanticUnits", []):
            reading = unit.get("reading", "")
            if reading and not contains_any_candidate(target, [reading]):
                missing.append(f"{unit.get('source')}â†’{reading}")

        if missing:
            critical.append(
                f"COMPOSITION_UNIT_LOSS {comp.get('sourceSpan')} "
                f"canonical={candidate}; missing={missing}"
            )
        else:
            warnings.append(
                f"COMPOSITION_ORDER_VARIANT {comp.get('sourceSpan')} "
                f"canonical={candidate}; semantic-units-preserved"
            )
    return warnings, critical


def compound_semantic_component_check(target, sino_compounds, morphemes, compound_policy, special_spans=None):
    """Semantic preservation for STRONG domain compounds without exact-text forcing."""
    warnings, critical = [], []
    by_span = {x.get("sourceSpan"): x for x in (special_spans or [])}
    targets = compound_policy.get("semanticComponentTargets", {})
    critical_cats = set(compound_policy.get("criticalModifierCategories", []))
    for comp in primary_sino_compounds(sino_compounds or []):
        if str(comp.get("semanticAuthority", comp.get("authority", "SOFT"))).upper() != "STRONG":
            continue
        special = by_span.get(comp.get("sourceSpan"), {})
        if special.get("targetTextAuthority"):
            continue
        if any(x.get("targetTextAuthority") and x.get("sourceSpan") and x.get("sourceSpan") in comp.get("sourceSpan", "") for x in (special_spans or [])):
            continue
        if special.get("candidateType") == "ORDINARY_OBJECT":
            continue
        missing = []
        for unit in comp.get("semanticUnits", [])[:-1]:
            src = unit.get("source", "")
            pieces = [src] if src in targets else list(src)
            for ch in pieces:
                entry = morphemes.get(ch, {})
                if entry.get("category") not in critical_cats and ch not in targets:
                    continue
                candidates = list(targets.get(ch) or [])
                if not candidates and entry.get("reading"):
                    candidates = [entry.get("reading")]
                if candidates and not contains_any_candidate(target, candidates):
                    missing.append(f"{ch}â†’{candidates}")
        if missing:
            critical.append(f"COMPOUND_SEMANTIC_COMPONENT_LOSS {comp.get('sourceSpan')} missing={missing}")
    return warnings, critical

def _count_nonoverlap_candidates(target, candidates):
    nt = norm_vi(target)
    spans = []
    for cand in sorted({norm_vi(x) for x in candidates if x}, key=len, reverse=True):
        if not cand:
            continue
        for m in re.finditer(re.escape(cand), nt):
            spans.append((m.start(), m.end()))
    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    chosen = []
    for st, en in spans:
        if any(not (en <= a or st >= b) for a, b in chosen):
            continue
        chosen.append((st, en))
    return len(chosen)


def compound_head_multiplicity_check(target, sino_compounds, compound_policy):
    """Prevent one target head mention from satisfying multiple source compounds."""
    warnings, critical = [], []
    groups = {}
    for comp in primary_sino_compounds(sino_compounds or []):
        if str(comp.get("semanticAuthority", comp.get("authority", "SOFT"))).upper() != "STRONG":
            continue
        anchor = comp.get("anchor", "")
        allowed = list((compound_policy.get("semanticAnchorTargets", {}) or {}).get(anchor, []))
        if not anchor or not allowed:
            continue
        groups.setdefault(anchor, {"count": 0, "allowed": allowed, "spans": []})
        groups[anchor]["count"] += 1
        groups[anchor]["spans"].append(comp.get("sourceSpan"))
    for anchor, g in groups.items():
        if g["count"] <= 1:
            continue
        found = _count_nonoverlap_candidates(target, g["allowed"])
        if found < g["count"]:
            critical.append(
                f"COMPOUND_HEAD_MULTIPLICITY_LOSS anchor={anchor} source_count={g['count']} target_count={found} spans={g['spans']}"
            )
    return warnings, critical


def structural_polarity_check(target, selected_senses):
    """Target-side check for structural polarity frames that loose token matching can mask."""
    warnings, critical = [], []
    nt = norm_vi(target)
    if any(s.get("senseId") == "double-negation-existence" for s in selected_senses or []):
        ok = any(x in nt for x in (
            "khÃ´ng pháº£i lÃ  khÃ´ng cÃ³", "khÃ´ng pháº£i khÃ´ng cÃ³", "khÃ´ng háº³n lÃ  khÃ´ng cÃ³",
            "váº«n cÃ³", "cÅ©ng cÃ³", "khÃ´ng pháº£i hoÃ n toÃ n khÃ´ng cÃ³"
        ))
        # 'cÅ©ng khÃ´ng cÃ³' / 'váº«n khÃ´ng cÃ³' are the opposite polarity.
        if "cÅ©ng khÃ´ng cÃ³" in nt or "váº«n khÃ´ng cÃ³" in nt:
            ok = False
        if not ok:
            critical.append("DOUBLE_NEGATION_EXISTENCE_LOSS")
    return warnings, critical


def style_register_check(target, sino_compounds, style_pack):
    warnings = []
    nt = norm_vi(target)
    modern = style_pack.get("modernLiteralWarnings", {})

    for comp in primary_sino_compounds(sino_compounds):
        candidate = comp.get("candidate", "")
        # Exact canonical-form warnings are meaningful only when the span has target-text authority.
        if comp.get("targetTextAuthority") and candidate and not contains_any_candidate(target, [candidate]):
            warnings.append(
                f"CANONICAL_SINO_COMPOUND_VARIANT {comp['sourceSpan']} â†’ candidate {candidate}"
            )

        for src_piece in comp.get("tokens", []):
            for literal in modern.get(src_piece, []):
                if norm_vi(literal) in nt:
                    warnings.append(
                        f"REGISTER_LITERALISM {comp['sourceSpan']} contains {src_piece}; "
                        f"target uses everyday literal '{literal}'"
                    )

    return list(dict.fromkeys(warnings))

def unsupported_addition_check(source, target, style_pack):
    warnings = []
    nt = norm_vi(target)
    for rule in style_pack.get("unsupportedTargetAdditions", []):
        target_phrase = rule.get("target", "")
        required = rule.get("requiresSourceAny", [])
        if target_phrase and norm_vi(target_phrase) in nt and not any(x in source for x in required):
            warnings.append(
                f"UNSUPPORTED_TARGET_ADDITION '{target_phrase}' requires source evidence {required}"
            )
    return warnings



def detect_semantic_sets(tokens, morphemes, matched_terms, selected_senses, compound_policy):
    """Detect generic structured enumerations separately from named compounds.

    This is data-driven from compound_policy.enumerationTokens and morpheme readings;
    it contains no benchmark sentence or expected translation mapping.
    """
    enum = set(compound_policy.get("enumerationTokens", []))
    min_run = int(compound_policy.get("enumerationMinRun", 3))
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] not in enum:
            i += 1
            continue
        j = i
        while j + 1 < len(tokens) and tokens[j + 1] in enum:
            j += 1
        run = tokens[i:j+1]
        if len(run) >= min_run:
            readings = []
            valid = True
            for tok in run:
                reading = morpheme_reading(tok, morphemes, matched_terms, selected_senses)
                if not reading:
                    valid = False
                    break
                readings.append(reading)
            if valid:
                out.append({
                    "sourceSpan": "".join(run),
                    "tokens": list(run),
                    "readings": readings,
                    "candidate": ", ".join(readings),
                    "class": "semantic_set",
                    "authority": "STRONG",
                    "startTokenIndex": i,
                    "endTokenIndex": j,
                    "sourceMutation": False,
                })
        i = j + 1
    return out


def _compact_guidance_items(terms, names, homophone_risks, sino_compounds, selected_senses, semantic_sets, analysis=None, special_spans=None, quantity_frames=None, noun_phrases=None, creature_rows=None):
    """Small guidance surface; semantic candidates never become blind text locks."""
    items = []
    analysis = analysis or {}; special_spans = special_spans or []; quantity_frames = quantity_frames or []; noun_phrases = noun_phrases or []; creature_rows = creature_rows or []
    if "fragment_or_nominal_ellipsis" in (analysis.get("features") or []):
        items.append(("grammar", "çœç•¥åè¯çŸ­è¯­ï¼šä¿æŒåè¯æ€§ï¼Œä¸æ–°å¢žåŽŸæ–‡æ²¡æœ‰çš„åŠ¨ä½œ"))

    for row in special_spans:
        span=row.get("sourceSpan",""); ctype=row.get("candidateType",""); canonical=row.get("canonicalTarget","")
        candidates=list(row.get("targetCandidates") or [])
        if canonical and canonical not in candidates: candidates.insert(0,canonical)
        if row.get("targetTextAuthority") and canonical:
            items.append(("special",f"{span} â†’ {canonical} (ä¸“å/å‘½åå®žä½“ï¼Œè¯‘åå›ºå®š)"))
        elif ctype=="IDIOM_FIXED_EXPRESSION" and candidates:
            items.append(("special",f"{span} â†’ {' / '.join(candidates[:3])} (å›ºå®šè¡¨è¾¾ï¼ŒæŒ‰æ•´ä½“å«ä¹‰)"))

    # FIX6.4.8: strong NP and creature rows are represented by the compressed semantic plan; report guidance stays diagnostic-only.

    # Strong domain compounds get semantic guidance but not exact surface authority.
    special_by_span={x.get("sourceSpan"):x for x in special_spans}
    for comp in primary_sino_compounds(sino_compounds):
        if str(comp.get("semanticAuthority",comp.get("authority","SOFT"))).upper()!="STRONG": continue
        if comp.get("targetTextAuthority"): continue
        sp=special_by_span.get(comp.get("sourceSpan"),{})
        if sp.get("targetTextAuthority"): continue
        if any(x.get("targetTextAuthority") and x.get("sourceSpan") and x.get("sourceSpan") in comp.get("sourceSpan", "") for x in special_spans):
            continue
        if sp.get("candidateType")=="ORDINARY_OBJECT": continue
        candidate=comp.get("candidate","")
        if candidate:
            items.append(("compound-semantic",f"{comp.get('sourceSpan')} â†’ {candidate} (è¯­ä¹‰å‚è€ƒï¼Œå¯è‡ªç„¶è¡¨è¾¾ï¼Œä½†ä¸è¦æ”¹å˜ç»„æˆå«ä¹‰)"))

    for qf in quantity_frames:
        if qf.get("role")!="DURATION_MEASURE": continue
        candidates=list(qf.get("targetCandidates") or [])
        if candidates:
            items.append(("quantity",f"{qf.get('sourceSpan')} â†’ {' / '.join(candidates[:3])} (æ—¶é•¿ï¼Œä¿ç•™æ•°é‡ä¸Žå•ä½)"))

    for sense in selected_senses:
        candidates=[]; preferred=sense.get("preferredTarget")
        if preferred: candidates.append(preferred)
        for c in sense.get("targetCandidates") or []:
            if c and c not in candidates: candidates.append(c)
        if candidates:
            role=sense.get("semanticRole") or "sense"; authority=str(sense.get("qaAuthority","SOFT")).upper()
            if authority=="STRONG":
                items.append(("sense",f"{sense.get('surface')} â†’ {' / '.join(candidates[:3])} ({role}; å·²é€‰æ‹©æ­¤ä¹‰)"))
            else:
                items.append(("sense",f"{sense.get('surface')} â†’ {candidates[0]} ({role})"))

    for comp in primary_sino_compounds(sino_compounds):
        if not comp.get("targetTextAuthority"): continue
        candidate=comp.get("candidate","")
        if candidate:
            head=comp.get("headReading",""); cls=comp.get("class",""); suffix=f"; head={head}" if head else ""
            items.append(("compound",f"{comp.get('sourceSpan')} â†’ {candidate} ({cls}{suffix})"))
    for item in semantic_sets:
        candidate=item.get("candidate","")
        if candidate: items.append(("set",f"{item.get('sourceSpan')} â†’ {candidate} (structured set)"))

    seen_src={x.split(" â†’ ",1)[0] for _,x in items if " â†’ " in x}
    for e in terms:
        src=e.get("source","").strip(); tgt=e.get("preferredTarget","").strip()
        if src and tgt and src not in seen_src: items.append(("term",f"{src} â†’ {tgt}"))
    for n in names:
        if n.get("surnameEvidence")=="EXACT_COMMON_SURNAME" and n.get("surnameVietnamese"):
            src=n.get("surnameSurface")
            if src not in seen_src: items.append(("name",f"{src} (surname) â†’ {n.get('surnameVietnamese')}"))
    if homophone_risks: items.append(("risk","homophone surname risk detected; keep Chinese source unchanged"))

    out=[]; seen=set()
    for kind,text in items:
        key=(kind,text.casefold())
        if key in seen: continue
        seen.add(key); out.append((kind,text))
        if len(out)>=4: break
    return out

def build_prompt(source, prev_text, next_text, terms, names, homophone_risks, analysis, sino_compounds, selected_senses, semantic_sets=None, special_spans=None, quantity_frames=None, noun_phrases=None, creature_rows=None):
    """FIX6.4.8: stable generation surface + at most one compact semantic-plan line.

    Raw PREV/NEXT are never sent. No analyzer JSON, prose contract, or expected sentence is sent.
    """
    semantic_sets=semantic_sets or []; special_spans=special_spans or []; quantity_frames=quantity_frames or []
    plan=build_semantic_plan(terms,special_spans,quantity_frames,noun_phrases or [],creature_rows or [],selected_senses)
    core="å°†ä¸‹é¢è¿™ä¸€å¥ä¸­æ–‡ç¿»è¯‘æˆè‡ªç„¶ã€å¿ å®žçš„è¶Šå—è¯­ã€‚ä¿ç•™å¦å®šã€æ•°é‡ã€äººç‰©å…³ç³»å’Œæ ¸å¿ƒå®žä½“ã€‚åªè¾“å‡ºè¶Šå—è¯­è¯‘æ–‡ï¼Œä¸è§£é‡Šã€‚"
    rendered=render_semantic_plan(plan)
    if not rendered:
        return core+"\n"+source
    return core+"\nè¯ä¹‰å‚è€ƒï¼š"+"ï¼›".join(rendered[:3])+"\n"+source

def translate(base: str, prompt: str, timeout=120, temperature=0.7):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "add_generation_prompt": False,
        "temperature": temperature,
        "top_k": 20,
        "top_p": 0.6,
        "repeat_penalty": 1.05,
        "seed": SEED,
        "max_tokens": 96,
        "stop": STOP,
        "stream": False,
    }

    res = http_json(base + "/v1/chat/completions", payload, timeout=timeout)
    choices = res.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"KhÃ´ng cÃ³ choices: {res!r}")

    msg = choices[0].get("message") or {}
    text = msg.get("content")
    if not isinstance(text, str):
        raise RuntimeError(f"KhÃ´ng cÃ³ message.content: {res!r}")

    return text.strip()

def norm_vi(s: str):
    return " ".join(unicodedata.normalize("NFC", s).casefold().split())


def candidate_evidence(target: str, candidate: str):
    nt = norm_vi(target)
    nc = norm_vi(candidate)

    if nc in nt:
        return True

    toks = [x for x in nc.split() if x]
    if len(toks) < 2:
        return False

    pos = 0
    for tok in toks:
        idx = nt.find(tok, pos)
        if idx < 0:
            return False
        pos = idx + len(tok)

    return True



def contains_any_candidate(target, candidates):
    return any(candidate_evidence(target, c) for c in candidates if c)


def semantic_set_check(target, semantic_sets):
    warnings, critical = [], []
    for item in semantic_sets or []:
        missing = [r for r in item.get("readings", []) if not contains_any_candidate(target, [r])]
        if missing:
            critical.append(
                f"SEMANTIC_SET_MEMBER_LOSS {item.get('sourceSpan')} missing={missing}"
            )
    return warnings, critical


def nominal_ellipsis_check(target, analysis):
    warnings, critical = [], []
    if "fragment_or_nominal_ellipsis" not in (analysis.get("features") or []):
        return warnings, critical
    nt = norm_vi(target).lstrip(" .,:;!?â€¦-")
    # Conservative predicate-start guard for a source that is structurally a noun phrase.
    invented_predicate_starts = (
        "bá» ", "Ä‘á»ƒ ", "giao ", "Ä‘Æ°a ", "mang ", "lÃ m ", "giá»¯ ",
        "dÃ¹ng ", "sá»­ dá»¥ng ", "chuyá»ƒn ", "Ä‘áº·t ", "hÃ£y ", "cáº§n ", "pháº£i "
    )
    if any(nt.startswith(x) for x in invented_predicate_starts):
        critical.append("ELLIPSIS_ADDED_PREDICATE")
    return warnings, critical




def _target_words(text: str):
    return [x for x in norm_vi(text).replace("\n", " ").split(" ") if x]


def negation_preservation_check(source, target, analysis):
    """Fail closed when explicit Chinese negation disappears from Vietnamese output.

    This is a generic meaning-frame guard, not a lexical target answer. It deliberately
    prefers review over silently accepting a polarity flip.
    """
    warnings, critical = [], []
    src_negs = list(analysis.get("negation") or [])
    if not src_negs:
        return warnings, critical
    nt = " " + norm_vi(target) + " "
    vi_negative_cues = (
        " khÃ´ng ", " chÆ°a ", " cháº³ng ", " cháº£ ", " Ä‘á»«ng ", " chá»› ", " khá»i ",
        " khÃ´ng thá»ƒ ", " khÃ´ng Ä‘Æ°á»£c ", " chÆ°a thá»ƒ ", " chÆ°a cháº¯c ", " khÃ´ng cÃ²n ",
        " khÃ´ng há» ", " vÃ´ ", " báº¥t ", " phi ", " miá»…n ", " cáº¥m ",
    )
    if not any(cue in nt for cue in vi_negative_cues):
        critical.append(f"NEGATION_PRESERVATION_LOSS source={src_negs}")
    return warnings, critical


def output_purity_check(source, target):
    """One subtitle in -> one Vietnamese subtitle out; no prompt/instruction leakage."""
    warnings, critical = [], []
    raw = target.strip()
    nt = norm_vi(raw)
    nonempty_lines = [x.strip() for x in raw.splitlines() if x.strip()]

    if len(nonempty_lines) > 1:
        critical.append("MULTILINE_OUTPUT")

    leak_fragments = (
        "giá»¯ láº¡i sá»± phá»§ Ä‘á»‹nh", "giá»¯ láº¡i cÃ¡c tá»«", "má»‘i quan há»‡ giá»¯a nhÃ¢n váº­t",
        "thÃ´ng tin liÃªn quan Ä‘áº¿n cÃ¢u hiá»‡n táº¡i", "cÃ¢u hiá»‡n táº¡i", "chá»‰ dÃ¹ng Ä‘á»ƒ hiá»ƒu",
        "chá»‰ xuáº¥t", "khÃ´ng giáº£i thÃ­ch", "thuáº­t ngá»¯ tham kháº£o", "tham kháº£o thuáº­t ngá»¯",
        "dá»‹ch cÃ¢u hiá»‡n táº¡i", "báº£n dá»‹ch tiáº¿ng viá»‡t", "source sentence", "current sentence",
        "relevant terms", "semantic role", "structured set", "head=", "source mutation",
    )
    if any(x in nt for x in leak_fragments) or "â†’" in raw:
        critical.append("PROMPT_METADATA_LEAK")

    # A single subtitle segment should not balloon into a mini-paragraph.
    src_len = max(1, cjk_count(source))
    words = _target_words(raw)
    sentence_ends = sum(raw.count(ch) for ch in ".!?ã€‚ï¼ï¼Ÿ")
    if len(words) > max(24, src_len * 3) or sentence_ends >= 3:
        critical.append("CONTEXT_OR_PROMPT_OVERGENERATION")

    return warnings, critical




def _zh_number_value(text):
    if text == "åŠ": return 0.5
    if text.endswith("åŠ") and text[:-1]:
        base=_zh_number_value(text[:-1])
        return None if base is None else base+0.5
    digits={"ã€‡":0,"é›¶":0,"ä¸€":1,"äºŒ":2,"ä¸¤":2,"ä¸‰":3,"å››":4,"äº”":5,"å…­":6,"ä¸ƒ":7,"å…«":8,"ä¹":9}
    units={"å":10,"ç™¾":100,"åƒ":1000,"ä¸‡":10000}
    if all(ch in digits for ch in text):
        if len(text)==1: return float(digits[text])
        try: return float(int(''.join(str(digits[ch]) for ch in text)))
        except Exception: return None
    total=0; section=0; number=0; seen=False
    for ch in text:
        if ch in digits:
            number=digits[ch]; seen=True
        elif ch in units:
            seen=True; unit=units[ch]
            if unit==10000:
                section=(section+number)*unit; total+=section; section=0; number=0
            else:
                if number==0: number=1
                section += number*unit; number=0
        else:
            return None
    return float(total+section+number) if seen else None


def _vi_number_word(value):
    if not float(value).is_integer(): return str(value).rstrip("0").rstrip(".")
    n=int(value)
    small={0:"khÃ´ng",1:"má»™t",2:"hai",3:"ba",4:"bá»‘n",5:"nÄƒm",6:"sÃ¡u",7:"báº£y",8:"tÃ¡m",9:"chÃ­n",10:"mÆ°á»i"}
    if n<=10: return small[n]
    if n<20: return "mÆ°á»i "+("lÄƒm" if n==15 else small[n-10])
    if n<100:
        t,r=divmod(n,10); out=small[t]+" mÆ°Æ¡i"
        if r: out += " "+("má»‘t" if r==1 else "lÄƒm" if r==5 else small[r])
        return out
    if n<1000:
        h,r=divmod(n,100); out=small[h]+" trÄƒm"
        if r:
            if r<10: out += " láº» "+small[r]
            else: out += " "+_vi_number_word(r)
        return out
    return str(n)


def _duration_measure_candidates(amount_text, unit, rule):
    value=_zh_number_value(amount_text.replace("ä¸ª","")); out=[]; unit_targets=list(rule.get("targetCandidates") or [])
    if value is None: return unit_targets
    base=rule.get("baseMinutes")
    if base is not None:
        minutes=float(base)*value
        if minutes%60==0:
            hours=int(minutes//60); out += (["má»™t giá»","1 giá»"] if hours==1 else [f"{_vi_number_word(hours)} giá»",f"{hours} giá»"])
        else:
            m=int(minutes) if minutes.is_integer() else minutes; out += [f"{_vi_number_word(m)} phÃºt",f"{m} phÃºt"]
    if amount_text.startswith("åŠ"):
        for t in unit_targets: out += [f"ná»­a {t}",f"má»™t ná»­a {t}"]
    elif value==1:
        for t in unit_targets: out += [f"má»™t {t}",t]
    else:
        nword=_vi_number_word(value)
        for t in unit_targets:
            if cjk_count(t)==0: out.append(f"{nword} {t}")
    return list(dict.fromkeys(x for x in out if x))


def detect_quantity_frames(source, tokens, special_policy, compound_policy):
    """Contextual quantity/duration frames with full Chinese-number spans.

    Cardinal spans contained inside a duration phrase are suppressed, so å…­ in å…­ä¸ªæ—¶è¾°
    cannot independently demand the Vietnamese word 'sÃ¡u' when the normalized duration is 12 giá».
    """
    q=(special_policy or {}).get("quantity",{}); frames=[]; duration_ranges=[]
    for src,cands in (q.get("strongQuantifiers") or {}).items():
        if src in source: frames.append({"sourceSpan":src,"role":"QUANTITY_BOUND","targetCandidates":list(cands),"authority":"STRONG"})

    num_pat=r"(?:[ã€‡é›¶ä¸€äºŒä¸¤ä¸‰å››äº”å…­ä¸ƒå…«ä¹åç™¾åƒä¸‡]{1,6}|åŠ)"
    duration_rules=q.get("durationUnitRules") or {}; covered=set()
    for unit,rule in sorted(duration_rules.items(),key=lambda kv:len(kv[0]),reverse=True):
        pat=re.compile(f"({num_pat})(ä¸ª)?({re.escape(unit)})")
        for m in pat.finditer(source):
            # å¹´ä»½ is a lexical noun, not a duration unit occurrence.
            if unit=='å¹´' and m.end(3)<len(source) and source[m.end(3):m.end(3)+1]=='ä»½':
                continue
            amount=m.group(1); cands=_duration_measure_candidates(amount,unit,rule)
            value=_zh_number_value(amount)
            frames.append({"sourceSpan":m.group(0),"role":"DURATION_MEASURE","targetCandidates":cands,"authority":"STRONG","unit":unit,"amount":amount,
                           "normalizedMinutes":None if rule.get("baseMinutes") is None or value is None else float(rule.get("baseMinutes"))*value,
                           "_start":m.start(),"_end":m.end()})
            duration_ranges.append((m.start(),m.end())); covered.add((unit,m.start(3)))

    for unit,cands in (q.get("durationUnits") or {}).items():
        if len(unit)<=1: continue
        start=0
        while True:
            pos=source.find(unit,start)
            if pos<0: break
            start=pos+len(unit)
            if (unit,pos) in covered: continue
            frames.append({"sourceSpan":unit,"role":"DURATION_UNIT","targetCandidates":list(cands),"authority":"STRONG","_start":pos,"_end":pos+len(unit)})

    for unit in ("å¤©","æ—¥"):
        cands=list((q.get("durationUnits") or {}).get(unit,[]))
        if not cands: continue
        pat=re.compile(f"({num_pat})(?:ä¸ª)?({unit})")
        for m in pat.finditer(source):
            if any(a<=m.start() and m.end()<=b for a,b in duration_ranges): continue
            frames.append({"sourceSpan":m.group(0),"role":"DURATION_MEASURE",
                           "targetCandidates":_duration_measure_candidates(m.group(1),unit,{"baseMinutes":1440,"targetCandidates":cands}),
                           "authority":"STRONG","unit":unit,"amount":m.group(1),"_start":m.start(),"_end":m.end()})
            duration_ranges.append((m.start(),m.end()))

    classifiers=sorted(set((compound_policy or {}).get("classifierBoundaryTokens",[])),key=len,reverse=True)
    # Longest Chinese-number sequence immediately before a classifier.
    classifier_alt='|'.join(re.escape(x) for x in classifiers if x)
    if classifier_alt:
        for m in re.finditer(f"({num_pat})(?=(?:{classifier_alt}))",source):
            st,en=m.start(1),m.end(1)
            if any(a<=st and en<=b for a,b in duration_ranges): continue
            val=_zh_number_value(m.group(1))
            if val is None: continue
            cands=[_vi_number_word(val)]
            if float(val).is_integer(): cands.append(str(int(val)))
            frames.append({"sourceSpan":m.group(1),"role":"CARDINAL_QUANTITY","targetCandidates":list(dict.fromkeys(cands)),"authority":"STRONG","_start":st,"_end":en})

    out=[]; seen=set()
    for row in frames:
        key=(row["sourceSpan"],row["role"],row.get('_start'))
        if key not in seen:
            seen.add(key)
            row=dict(row); row.pop('_start',None); row.pop('_end',None); out.append(row)
    return out

def quantity_frame_check(target, quantity_frames):
    warnings, critical = [], []
    for frame in quantity_frames or []:
        candidates = frame.get("targetCandidates") or []
        if candidates and not contains_any_candidate(target, candidates):
            msg = f"QUANTITY_FRAME_LOSS {frame.get('sourceSpan')} role={frame.get('role')} expected={candidates}"
            (critical if frame.get("authority") == "STRONG" else warnings).append(msg)
    return warnings, critical


def _titlecase_runs(text):
    raw_tokens = re.findall(r"[^\s]+", text, flags=re.UNICODE)
    cleaned = []
    for raw in raw_tokens:
        token = raw.strip(".,!?;:()[]{}\"'â€œâ€â€˜â€™â€¦-â€”")
        cleaned.append(token)
    runs, cur = [], []
    for tok in cleaned:
        letters = "".join(ch for ch in tok if ch.isalpha())
        titleish = bool(letters) and letters[0].isupper() and all(ch.isalpha() or ch in "-'" for ch in tok)
        if titleish:
            cur.append(tok)
        else:
            if len(cur) >= 2:
                runs.append(" ".join(cur))
            cur = []
    if len(cur) >= 2:
        runs.append(" ".join(cur))
    return runs


def _source_morpheme_readings(source, morphemes):
    out=[]
    for ch in source:
        e=morphemes.get(ch) if is_cjk(ch) else None
        if e and e.get("reading"): out.append(norm_vi(e.get("reading")))
    return out


def _run_supported_by_source_morphemes(run, source, morphemes):
    words=[norm_vi(x.strip(".,!?;:()[]{}\\\"'â€œâ€â€˜â€™â€¦-â€”")) for x in run.split()]; words=[x for x in words if x]
    readings=_source_morpheme_readings(source,morphemes)
    if not words or not readings: return False
    pos=0
    for w in words:
        found=False
        for i in range(pos,len(readings)):
            if w==readings[i] or w in readings[i] or readings[i] in w:
                pos=i+1; found=True; break
        if not found: return False
    return True

def hallucinated_entity_check(source, target, special_spans, names, matched_terms, sino_compounds, morphemes):
    """Reject opaque invented proper names, not source-derived HÃ¡n-Viá»‡t/descriptive capitalization."""
    warnings,critical=[],[]; allowed=[]
    for row in special_spans or []: allowed += [row.get("canonicalTarget","")] + list(row.get("targetCandidates") or [])
    for n in names or []: allowed.append(n.get("surnameVietnamese",""))
    for e in matched_terms or []: allowed += [e.get("preferredTarget","")] + list(e.get("targetCandidates") or [])
    for c in sino_compounds or []: allowed.append(c.get("candidate",""))
    allowed_norm=[norm_vi(x) for x in allowed if x]
    for run in _titlecase_runs(target):
        nr=norm_vi(run)
        if any(nr in a or a in nr for a in allowed_norm if a): continue
        if _run_supported_by_source_morphemes(run,source,morphemes): continue
        toks=run.split(); opaque_ascii=all(all(ord(ch)<128 for ch in tok) for tok in toks); suspicious=any(re.search(r"[jqwz]",tok.casefold()) for tok in toks)
        if len(toks)>=2 and opaque_ascii and suspicious:
            critical.append(f"HALLUCINATED_ENTITY_CANDIDATE '{run}'")
    return warnings,critical

def source_sovereignty_check(source, target, analysis):
    """Structural invariants that must come from CURRENT, independent of PREV/NEXT."""
    warnings, critical = [], []
    # Current-source polarity is the strongest cheap invariant.
    nw, nc = negation_preservation_check(source, target, analysis)
    warnings.extend(nw); critical.extend(nc)
    return warnings, critical

def post_check(source, target, matched_terms, names, homophone_risks, analysis, sino_compounds, style_pack, selected_senses, compound_policy, semantic_sets=None, special_spans=None, quantity_frames=None, morphemes=None, noun_phrases=None, creature_rows=None, creature_policy=None, semantic_plan=None, meaning_frame=None):
    warnings = []
    critical = []

    stripped = target.strip()
    nt = norm_vi(stripped)

    if not stripped:
        critical.append("EMPTY_TARGET")
    if stripped == source.strip():
        critical.append("SOURCE_ECHO")
    if cjk_count(stripped) > 0:
        critical.append("NON_VIETNAMESE_CJK_OUTPUT")

    leakage_markers = [
        "ç¿»è¯‘ä¸ºä¸­æ–‡","ç¿»è¯‘æˆä¸­æ–‡","ä¸Šä¸€å¥","ä¸‹ä¸€å¥","å½“å‰å¥åˆ†è¯","è¯­æ³•çº¿ç´¢",
        "åŽç»­è¯­ä¹‰é¢„æœŸ","æœ¯è¯­å‚è€ƒ","ä¸“åä¿æŠ¤","åŒéŸ³é£Žé™©","è¯ä¹‰åˆ¤åˆ«",
        "å—ä¿æŠ¤çš„å¤šè¯æ±‰è¶Šç»„åˆ","å½“å‰å¥ç›¸å…³å‚è€ƒ","å½“å‰å¥æœ¯è¯­å‚è€ƒ",
        "å‰æ–‡ï¼ˆåªç”¨äºŽç†è§£ï¼‰","åŽæ–‡ï¼ˆåªç”¨äºŽç†è§£ï¼‰","å½“å‰å¥ï¼š","ã€å½“å‰å¥ã€‘",
        "cÃ¢u nguá»“n","giáº£i thÃ­ch:"
    ]
    for marker in leakage_markers:
        if norm_vi(marker) in nt:
            critical.append(f"PROMPT_OR_CONTEXT_LEAK:{marker}")

    if cjk_count(source) <= 4 and len(stripped) > 100:
        critical.append("OVER_GENERATION")

    for e in matched_terms:
        candidates = list(e.get("targetCandidates") or [])
        preferred = e.get("preferredTarget")
        if preferred and preferred not in candidates:
            candidates.insert(0, preferred)
        if candidates and not contains_any_candidate(stripped, candidates):
            if bool(e.get("targetTextAuthority", False)):
                critical.append(
                    f"CANONICAL_TERM_LOSS {e.get('source')} canonical={preferred or candidates[0]}"
                )
            elif bool(e.get("criticalForQa", False)):
                critical.append(
                    f"TERM_SEMANTIC_LOSS {e.get('source')} expected={candidates}"
                )
            else:
                warnings.append(f"TERM_VARIANT {e.get('source')} â†’ suggestion {candidates}")

    for sense in selected_senses:
        candidates = list(sense.get("targetCandidates") or [])
        preferred = sense.get("preferredTarget")
        if preferred and preferred not in candidates:
            candidates.insert(0, preferred)
        authority = str(sense.get("qaAuthority", "SOFT")).upper()

        if candidates and not contains_any_candidate(stripped, candidates):
            msg = (
                f"SENSE_PRESERVATION_LOSS {sense.get('surface')} "
                f"sense={sense.get('senseId')} expected={candidates}"
            )
            (critical if authority == "STRONG" else warnings).append(msg)

        for bad in sense.get("forbiddenTargetPatterns") or []:
            if norm_vi(bad) in nt:
                msg = (
                    f"SENSE_CONTRADICTION {sense.get('surface')} "
                    f"sense={sense.get('senseId')} target-pattern={bad}"
                )
                (critical if authority == "STRONG" else warnings).append(msg)

    if homophone_risks:
        for h in homophone_risks:
            critical.append(
                "HOMOPHONE_SURNAME_SOURCE_REVIEW "
                f"{h['sourceToken']} ~ {h['candidateSurname']} "
                f"({h['pinyin']}); SOURCE_NOT_MUTATED"
            )

    for n in names:
        title = n.get("titleSource")
        title_terms = [e for e in matched_terms if e.get("source") == title]
        for e in title_terms:
            candidates = list(e.get("targetCandidates") or [])
            preferred = e.get("preferredTarget")
            if preferred and preferred not in candidates:
                candidates.insert(0, preferred)
            if candidates and not contains_any_candidate(stripped, candidates):
                warnings.append(f"TITLE_ROLE_LOSS {title} â†’ {candidates}")

        if (
            n.get("surnameEvidence") == "EXACT_COMMON_SURNAME"
            and n.get("surnameVietnamese")
            and not contains_any_candidate(stripped, [n.get("surnameVietnamese")])
        ):
            warnings.append(
                f"NAME_RENDERING_VARIANT {n.get('surnameSurface')} â†’ "
                f"recommended {n.get('surnameVietnamese')}"
            )

    ssw, ssc = special_span_check(stripped, special_spans or [])
    warnings.extend(ssw)
    critical.extend(ssc)

    qfw, qfc = quantity_frame_check(stripped, quantity_frames or [])
    warnings.extend(qfw)
    critical.extend(qfc)

    npw, npc = noun_phrase_semantic_check(stripped, noun_phrases or [])
    warnings.extend(npw)
    critical.extend(npc)

    crw, crc = creature_semantic_check(stripped, creature_rows or [], creature_policy or {})
    warnings.extend(crw)
    critical.extend(crc)

    plw, plc = semantic_plan_coverage_check(stripped, semantic_plan or [])
    warnings.extend(plw); critical.extend(plc)

    mfw, mfc = meaning_frame_coverage_check(stripped, meaning_frame or {})
    warnings.extend(mfw); critical.extend(mfc)

    hew, hec = hallucinated_entity_check(
        source, stripped, special_spans or [], names, matched_terms, sino_compounds, morphemes or {}
    )
    warnings.extend(hew)
    critical.extend(hec)

    warnings.extend(style_register_check(stripped, sino_compounds, style_pack))
    warnings.extend(unsupported_addition_check(source, stripped, style_pack))

    cw, cc = compound_composition_check(stripped, sino_compounds, compound_policy)
    warnings.extend(cw)
    critical.extend(cc)

    mcw, mcc = compound_semantic_component_check(stripped, sino_compounds, morphemes or {}, compound_policy, special_spans or [])
    warnings.extend(mcw); critical.extend(mcc)

    hmw, hmc = compound_head_multiplicity_check(stripped, sino_compounds, compound_policy)
    warnings.extend(hmw); critical.extend(hmc)

    spw, spc = structural_polarity_check(stripped, selected_senses)
    warnings.extend(spw); critical.extend(spc)

    aw, ac = compound_anchor_check(stripped, sino_compounds, compound_policy)
    warnings.extend(aw)
    critical.extend(ac)

    sw, sc = semantic_set_check(stripped, semantic_sets or [])
    warnings.extend(sw)
    critical.extend(sc)

    ew, ec = nominal_ellipsis_check(stripped, analysis)
    warnings.extend(ew)
    critical.extend(ec)

    pw, pc = output_purity_check(source, stripped)
    warnings.extend(pw)
    critical.extend(pc)

    qw, qc = source_sovereignty_check(source, stripped, analysis)
    warnings.extend(qw)
    critical.extend(qc)

    warnings = list(dict.fromkeys(warnings))
    critical = list(dict.fromkeys(critical))

    if critical:
        status = "NEEDS_REVIEW"
    elif warnings:
        status = "ACCEPTED_WITH_WARNING"
    else:
        status = "ACCEPTED"

    return status, warnings, critical

def main():
    input_path = HERE / "WEAK-300-CORPUS-FIX648.txt"
    sources = [
        x.strip()
        for x in input_path.read_text(encoding="utf-8-sig").splitlines()
        if x.strip()
    ]

    if len(sources) != 300:
        eprint(f"[STOP] Input pháº£i Ä‘Ãºng 300 cÃ¢u, hiá»‡n cÃ³ {len(sources)}.")
        return 2

    terms = load_terms()
    surnames = load_surnames()
    grammar = load_grammar()
    morphemes = load_sino_morphemes()
    compound_policy = load_compound_policy()
    style_pack = load_classical_style()
    sense_rules = load_contextual_senses()
    special_policy = load_special_span_policy()
    noun_policy = load_noun_phrase_policy()
    creature_policy = load_creature_policy()
    entity_registry = SessionEntityRegistry()

    term_index = build_alias_index(terms)
    surname_exact, surname_pinyin2 = build_surname_maps(surnames)

    exe = find_server()
    if not exe:
        eprint("[STOP] KhÃ´ng tÃ¬m tháº¥y llama-server.exe.")
        return 3

    base = f"http://127.0.0.1:{PORT}"
    proc = log = None

    try:
        print(f"[INFO] llama-server: {exe}", flush=True)
        print(f"[INFO] model: {MODEL}", flush=True)
        print(f"[INFO] xianxia terms: {len(terms)}", flush=True)
        print(f"[INFO] chinese surnames: {len(surnames)}", flush=True)
        print(f"[INFO] pypinyin homophone check: {'ON' if HAS_PYPINYIN else 'OFF'}", flush=True)
        print("[INFO] grammar segmentation: ON", flush=True)
        print(f"[INFO] sino-vietnamese morphemes: {len(morphemes)}", flush=True)
        print("[INFO] sino-vietnamese compound resolver: ON", flush=True)
        print("[INFO] classical Vietnamese style QA: ON", flush=True)
        print(f"[INFO] contextual sense rules: {len(sense_rules)}", flush=True)
        print("[INFO] compound classifier-boundary gate: ON", flush=True)
        print("[INFO] semantic-head completeness QA: ON", flush=True)
        print("[INFO] exact-surname rendering QA: ON", flush=True)
        print("[INFO] multi-token composition graph: ON", flush=True)
        print("[INFO] strong 3+char + evidence-based 2-char domain preservation: ON", flush=True)
        print("[INFO] compact stable guidance: ON", flush=True)
        print("[INFO] nominal-ellipsis semantic guard: ON", flush=True)
        print("[INFO] structural semantic frames: ON", flush=True)
        print("[INFO] evidence-based sense authority escalation: ON", flush=True)
        print("[INFO] semantic-set preservation: ON", flush=True)
        print("[INFO] WHOLE-SPAN before word-by-word resolver: ON", flush=True)
        print("[INFO] head-first noun-phrase semantic composer: ON", flush=True)
        print("[INFO] creature/ecology semantic composer: ON", flush=True)
        print("[INFO] semantic plan compression: ON (max 3 anchors)", flush=True)
        print("[INFO] semantic span ownership: named > quantity > creature > noun phrase > compound", flush=True)
        print("[INFO] named-parent dominance over contained ordinary/domain spans: ON", flush=True)
        print("[INFO] semanticAuthority / targetTextAuthority split: ON", flush=True)
        print("[INFO] quantity / duration meaning frames: ON", flush=True)
        print("[INFO] scoped polarity + A-not-A question resolver: ON", flush=True)
        print("[INFO] contextual classifier boundary: ON", flush=True)
        print("[INFO] strong compound semantic-component QA: ON", flush=True)
        print("[INFO] source-bound hallucinated-entity guard: ON", flush=True)
        print("[INFO] hallucinated-entity guard: ON", flush=True)
        print("[INFO] per-run entity registry: ON (EPHEMERAL ONLY; no persistence)", flush=True)
        print("[INFO] current-source sovereignty: ON (raw PREV/NEXT not sent to model)", flush=True)
        print("[INFO] output purity / single-line guard: ON", flush=True)
        print("[INFO] negation preservation guard: ON", flush=True)
        print("[INFO] second-pass repair: OFF", flush=True)
        print("[INFO] protected render slots: OFF", flush=True)
        print("[INFO] prev/current/next contextual evidence: RULE-RESOLVER ONLY", flush=True)
        print("[INFO] literal next-sentence prediction: OFF", flush=True)
        print("[INFO] next semantic expectation: ON", flush=True)
        print("[INFO] source mutation: OFF", flush=True)
        print(f"[INFO] seed={SEED}", flush=True)

        proc, log, _ = start_server(exe)
        wait_server(base, proc)
        print("[PASS] Server sáºµn sÃ ng.", flush=True)

        rows = []
        t0 = time.perf_counter()

        for i, source in enumerate(sources):
            prev_text = sources[i-1] if i > 0 else ""
            next_text = sources[i+1] if i + 1 < len(sources) else ""

            matched = retrieve_terms(source, term_index)
            title_terms = [
                e for e in terms
                if e.get("category") in {"honorific", "relationship-title", "sect-role"}
            ]
            names, homophones = detect_names(
                source, title_terms, surname_exact, surname_pinyin2, grammar
            )

            lexicon = make_segmentation_lexicon(terms, surnames, grammar, names, sense_rules)
            tokens = segment_chinese(source, lexicon)
            analysis = analyze_grammar(source, tokens, grammar, special_policy)
            selected_senses = resolve_contextual_senses(
                source, prev_text, next_text, tokens, sense_rules
            )
            sino_compounds = detect_sino_compounds(
                source, tokens, matched, morphemes, grammar, compound_policy, selected_senses
            )
            semantic_sets = detect_semantic_sets(
                tokens, morphemes, matched, selected_senses, compound_policy
            )
            structural_senses = detect_structural_semantic_frames(
                source, tokens, analysis, sino_compounds
            )
            selected_senses = merge_senses(selected_senses, structural_senses)
            special_spans = detect_special_spans(
                source, matched, names, sino_compounds, morphemes, special_policy, entity_registry
            )
            quantity_frames = detect_quantity_frames(
                source, tokens, special_policy, compound_policy
            )
            creature_rows = detect_creature_semantics(
                source, special_spans, creature_policy
            )
            noun_phrases = detect_noun_phrase_semantics(
                source, special_spans, noun_policy
            )
            sino_compounds, noun_phrases = suppress_lower_semantic_spans(
                sino_compounds, noun_phrases, creature_rows, quantity_frames, special_spans
            )
            semantic_plan = build_semantic_plan(
                matched, special_spans, quantity_frames, noun_phrases, creature_rows, selected_senses
            )
            meaning_frame = build_meaning_frame(
                source, tokens, analysis, special_spans, quantity_frames, noun_phrases,
                creature_rows, selected_senses, matched, structural_senses
            )

            prompt = build_prompt(
                source, prev_text, next_text, matched, names, homophones, analysis,
                sino_compounds, selected_senses, semantic_sets, special_spans, quantity_frames, noun_phrases, creature_rows
            )

            row_id = i + 1
            print(f"[{row_id:03d}/300] {source}", flush=True)
            print("        TOKENS: " + " | ".join(tokens), flush=True)
            print(
                "        GRAMMAR: "
                + (", ".join(analysis["features"]) if analysis["features"] else "(none)"),
                flush=True,
            )
            if analysis["nextSemanticExpectation"]:
                print(
                    "        EXPECT: "
                    + ", ".join(analysis["nextSemanticExpectation"]),
                    flush=True,
                )
            print(
                "        CONTEXT: "
                + ("prev/current/next" if analysis["contextNeeded"] else "not-needed"),
                flush=True,
            )
            if names:
                print(
                    "        NAME: "
                    + ", ".join(
                        f"{n['nameSource']}[{n['surnameEvidence']}]"
                        for n in names
                    ),
                    flush=True,
                )
            if homophones:
                print(
                    "        HOMO: "
                    + ", ".join(
                        f"{h['sourceToken']}~{h['candidateSurname']}({h['pinyin']})"
                        for h in homophones
                    ),
                    flush=True,
                )

            if selected_senses:
                print(
                    "        SENSE: "
                    + ", ".join(
                        f"{x['surface']}â†’{x.get('senseId')}[{x.get('qaAuthority')}]"
                        for x in selected_senses
                    ),
                    flush=True,
                )

            if sino_compounds:
                print(
                    "        SINO: "
                    + ", ".join(
                        f"{c['sourceSpan']}â†’{c['candidate']}[{c['class']}]"
                        for c in sino_compounds
                    ),
                    flush=True,
                )
            if semantic_sets:
                print(
                    "        SET: " + ", ".join(
                        f"{x['sourceSpan']}â†’{x['candidate']}" for x in semantic_sets
                    ),
                    flush=True,
                )
            if special_spans:
                print(
                    "        SPECIAL: " + ", ".join(
                        f"{x['sourceSpan']}[{x.get('candidateType')}|sem={x.get('semanticAuthority')}|text={x.get('targetTextAuthority')}]"
                        for x in special_spans
                    ),
                    flush=True,
                )
            if quantity_frames:
                print(
                    "        QUANTITY: " + ", ".join(
                        f"{x['sourceSpan']}[{x.get('role')}]" for x in quantity_frames
                    ),
                    flush=True,
                )

            if noun_phrases:
                print(
                    "        NP: " + ", ".join(
                        f"{x['sourceSpan']}[head={x.get('headSource')}]" for x in noun_phrases
                    ),
                    flush=True,
                )
            if creature_rows:
                print(
                    "        CREATURE: " + ", ".join(
                        f"{x['sourceSpan']}[{x.get('creatureType')}|head={x.get('headSource')}]" for x in creature_rows
                    ),
                    flush=True,
                )
            if semantic_plan:
                print("        PLAN: " + " || ".join(render_semantic_plan(semantic_plan)), flush=True)
            frame_nonempty = [k for k, v in meaning_frame.items() if v]
            if frame_nonempty:
                print("        FRAME: " + ", ".join(frame_nonempty), flush=True)

            started = time.perf_counter()
            try:
                target = translate(base, prompt)
                elapsed = time.perf_counter() - started
                status, warnings, critical = post_check(
                    source, target, matched, names, homophones, analysis,
                    sino_compounds, style_pack, selected_senses, compound_policy, semantic_sets,
                    special_spans, quantity_frames, morphemes, noun_phrases, creature_rows, creature_policy, semantic_plan, meaning_frame
                )

                rows.append({
                    "id": row_id,
                    "previousSource": prev_text,
                    "source": source,
                    "nextSource": next_text,
                    "tokens": tokens,
                    "grammarAnalysis": analysis,
                    "matchedKnowledge": [
                        {
                            "source": e.get("source"),
                            "preferredTarget": e.get("preferredTarget"),
                            "category": e.get("category"),
                        }
                        for e in matched
                    ],
                    "nameCandidates": names,
                    "homophoneSurnameRisks": homophones,
                    "contextualSenses": selected_senses,
                    "sinoVietnameseCompounds": sino_compounds,
                    "semanticSets": semantic_sets,
                    "specialSpans": special_spans,
                    "quantityFrames": quantity_frames,
                    "nounPhrases": noun_phrases,
                    "creatureSemantics": creature_rows,
                    "semanticPlan": semantic_plan,
                    "meaningFrame": meaning_frame,
                    "guidanceItems": [x[1] for x in _compact_guidance_items(
                        matched, names, homophones, sino_compounds, selected_senses, semantic_sets, analysis, special_spans, quantity_frames, noun_phrases, creature_rows
                    )],
                    "target": target,
                    "status": status,
                    "warnings": warnings,
                    "critical": critical,
                    "elapsed_s": elapsed,
                    "ok": True,
                })

                print(f"        â†’ {target}", flush=True)
                print(f"        [{status}] {elapsed:.3f}s", flush=True)
                for w in warnings:
                    print(f"        WARN: {w}", flush=True)
                for c in critical:
                    print(f"        CRIT: {c}", flush=True)

            except Exception as exc:
                elapsed = time.perf_counter() - started
                rows.append({
                    "id": row_id,
                    "previousSource": prev_text,
                    "source": source,
                    "nextSource": next_text,
                    "tokens": tokens,
                    "grammarAnalysis": analysis,
                    "matchedKnowledge": [],
                    "nameCandidates": names,
                    "homophoneSurnameRisks": homophones,
                    "contextualSenses": selected_senses,
                    "sinoVietnameseCompounds": sino_compounds,
                    "semanticSets": semantic_sets,
                    "specialSpans": special_spans,
                    "quantityFrames": quantity_frames,
                    "nounPhrases": noun_phrases,
                    "creatureSemantics": creature_rows,
                    "semanticPlan": semantic_plan,
                    "meaningFrame": meaning_frame,
                    "guidanceItems": [],
                    "target": "",
                    "status": "RUNTIME_ERROR",
                    "warnings": [],
                    "critical": [str(exc)],
                    "elapsed_s": elapsed,
                    "ok": False,
                })
                eprint(f"[STOP] CÃ¢u {row_id} lá»—i: {exc}")
                break

        total = time.perf_counter() - t0

        report = [
            "HY-MT1.5-1.8B + XIANXIA â€” SEMANTIC PLAN / WEAKNESS COVERAGE FIX6.4.9-A",
            "=" * 88,
            f"Model: {MODEL}",
            "Corpus: 300-line weakness-regression runtime (derived from FIX647 failure families; NEW sentences; single model session / single run)",
            f"Xianxia terms: {len(terms)}",
            f"Chinese surnames: {len(surnames)}",
            f"Pypinyin homophone check: {'ON' if HAS_PYPINYIN else 'OFF'}",
            "Grammar segmentation: ON",
            f"Sino-Vietnamese morphemes: {len(morphemes)}",
            "Sino-Vietnamese compound resolver: ON",
            "Classical Vietnamese style QA: ON",
            f"Contextual sense rules: {len(sense_rules)}",
            "Compound classifier-boundary gate: ON",
            "Semantic-head completeness QA: ON",
            "Exact-surname rendering QA: ON",
            "Multi-token composition graph: ON",
            "Strong 3+char + evidence-based 2-char domain preservation: ON",
            "Compact report guidance: ON (NOT injected as verbose prompt metadata)",
            "Nominal-ellipsis semantic guard: ON",
            "Structural semantic frames: ON",
            "Evidence-based sense authority escalation: ON",
            "Semantic-set preservation: ON",
            "WHOLE-SPAN before word-by-word resolver: ON",
            "Head-first noun-phrase semantic composer: ON",
            "Creature/ecology + body/trace semantic composer: ON",
            "Semantic plan compression: ON (max 3 anchors)",
            "Semantic span ownership: named > quantity > creature > noun phrase > compound",
            "Named-parent dominance over contained NP fragments: ON",
            "semanticAuthority / targetTextAuthority split: ON",
            "Quantity / duration meaning frames: ON",
            "Lexical-role-aware negation: ON",
            "Hallucinated-entity guard: ON",
            "Per-run entity registry: ON (PERSON_NAME requires fresh current-source name evidence; persistence OFF)",
            "Current-source sovereignty: ON",
            "Output purity / single-line guard: ON",
            "Negation preservation guard: ON",
            "Data-driven surname/title morphology + lexical false-name guard: ON",
            "Second-pass repair: OFF",
            "Protected render slots: OFF",
            "Context: PREV/NEXT used only by deterministic sense resolver; raw context not sent to model",
            "Literal next-source prediction: OFF",
            "Next semantic expectation: ON",
            "Source mutation: OFF",
            "Series Memory persistence: OFF",
            "Sentence-specific target hardcode: OFF",
            f"Seed: {SEED}",
            f"Total time: {total:.3f}s",
            f"Average: {total/max(len(rows),1):.3f}s/segment",
            "",
        ]

        vi_lines = []

        for r in rows:
            report += [
                f"[{r['id']:03d}] PREVIOUS", r["previousSource"] or "(none)",
                f"[{r['id']:03d}] SOURCE", r["source"],
                f"[{r['id']:03d}] NEXT", r["nextSource"] or "(none)",
                f"[{r['id']:03d}] TOKENS", " | ".join(r["tokens"]),
                f"[{r['id']:03d}] GRAMMAR", json.dumps(r["grammarAnalysis"], ensure_ascii=False),
                f"[{r['id']:03d}] KNOWLEDGE",
                ", ".join(
                    f"{k['source']}â†’{k['preferredTarget']}"
                    for k in r["matchedKnowledge"]
                ) or "(none)",
                f"[{r['id']:03d}] NAME", json.dumps(r["nameCandidates"], ensure_ascii=False),
                f"[{r['id']:03d}] HOMOPHONE", json.dumps(r["homophoneSurnameRisks"], ensure_ascii=False),
                f"[{r['id']:03d}] CONTEXTUAL_SENSE", json.dumps(r.get("contextualSenses", []), ensure_ascii=False),
                f"[{r['id']:03d}] SINO_COMPOUND", json.dumps(r["sinoVietnameseCompounds"], ensure_ascii=False),
                f"[{r['id']:03d}] SEMANTIC_SET", json.dumps(r.get("semanticSets", []), ensure_ascii=False),
                f"[{r['id']:03d}] SPECIAL_SPAN", json.dumps(r.get("specialSpans", []), ensure_ascii=False),
                f"[{r['id']:03d}] QUANTITY_FRAME", json.dumps(r.get("quantityFrames", []), ensure_ascii=False),
                f"[{r['id']:03d}] NOUN_PHRASE", json.dumps(r.get("nounPhrases", []), ensure_ascii=False),
                f"[{r['id']:03d}] CREATURE", json.dumps(r.get("creatureSemantics", []), ensure_ascii=False),
                f"[{r['id']:03d}] SEMANTIC_PLAN", json.dumps(r.get("semanticPlan", []), ensure_ascii=False),
                f"[{r['id']:03d}] GUIDANCE", json.dumps(r.get("guidanceItems", []), ensure_ascii=False),
                f"[{r['id']:03d}] TARGET", r["target"],
                f"[{r['id']:03d}] STATUS {r['status']}",
            ]

            for w in r["warnings"]:
                report.append(f"WARN: {w}")
            for c in r["critical"]:
                report.append(f"CRIT: {c}")

            report += [f"elapsed={r['elapsed_s']:.3f}s", "-" * 88]
            vi_lines.append(f"{r['id']:03d}. [{r['status']}] {r['target']}")

        (HERE / "WEAK300-FIX649A-result.txt").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )
        (HERE / "WEAK300-FIX649A-vietnamese.txt").write_text(
            "\n".join(vi_lines) + "\n", encoding="utf-8"
        )
        (HERE / "WEAK300-FIX649A-result.json").write_text(
            json.dumps({
                "model": MODEL,
                "grammarSegmentation": True,
                "sinoVietnameseCompoundResolver": True,
                "classicalVietnameseStyleQa": True,
                "contextualSenseResolver": True,
                "contextualSenseRuleCount": len(sense_rules),
                "compoundClassifierBoundaryGate": True,
                "semanticHeadCompletenessQa": True,
                "exactSurnameRenderingQa": True,
                "multiTokenCompositionGraph": True,
                "strongCompoundPreservation": True,
                "compactStableGuidance": True,
                "nominalEllipsisSemanticGuard": True,
                "structuralSemanticFrames": True,
                "evidenceBasedSenseAuthorityEscalation": True,
                "semanticSetPreservation": True,
                "wholeSpanBeforeWordByWord": True,
                "headFirstNounPhraseSemantics": True,
                "creatureEcologySemanticComposer": True,
                "semanticPlanCompression": True,
                "semanticSpanOwnership": "named>quantity>creature>noun-phrase>compound",
                "specialSpanResolver": True,
                "semanticVsTargetTextAuthority": True,
                "quantityDurationFrames": True,
                "lexicalRoleAwareNegation": True,
                "hallucinatedEntityGuard": True,
                "sessionEntityRegistryPersistence": False,
                "oneShotCompoundRepair": False,
                "protectedRenderSlots": False,
                "contextPolicy": "resolver-only-prev-next-raw-context-off",
                "literalNextPrediction": False,
                "nextSemanticExpectation": True,
                "sourceMutation": False,
                "seriesMemoryPersistence": False,
                "sentenceSpecificHardcode": False,
                "pypinyinAvailable": HAS_PYPINYIN,
                "rows": rows,
                "total_s": total,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )

        runtime_pass = len(rows) == 300 and all(r["ok"] for r in rows)
        if runtime_pass:
            print(f"[DONE] 300/300 runtime trong {total:.3f}s.", flush=True)
            print("[RESULT] WEAK300-FIX649A-result.txt", flush=True)
            print("[VI] WEAK300-FIX649A-vietnamese.txt", flush=True)
            return 0

        eprint("[FAIL] Runtime chÆ°a hoÃ n táº¥t Ä‘á»§ 300 cÃ¢u.")
        return 5

    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=8)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if log is not None:
            try:
                log.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

