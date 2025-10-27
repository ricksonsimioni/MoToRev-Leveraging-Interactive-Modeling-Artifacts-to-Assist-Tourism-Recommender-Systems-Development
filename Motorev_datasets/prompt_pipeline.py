"""
Ecore → XMI pipeline (XMI-only, conforms to the *domain* Ecore metamodel)

- Parses Ecore (.ecore) for the DOMAIN model (and optional RECOMMENDER model for grounding).
- Builds a low-hallucination prompt that forces STRICT XMI output (valid XML, EMF-style).
- Generates the XMI via an Ollama-compatible endpoint (e.g., gpt-oss cloud through Ollama).
- Validates the produced XMI *against the domain metamodel* with static checks (no external deps).
- Exports to disk as .xmi

Drop-in: save this as a standalone module.
"""

import os
import re
import json
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

# =============================================================================
# Core helpers
# =============================================================================

def _escape_braces(text: str) -> str:
    if text is None:
        return ""
    return text.replace("{", "{{").replace("}", "}}")

def _safe_slug(text: str, fallback: str = "artifact") -> str:
    if not text:
        return fallback
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or fallback

def _normalize_ext(file_extension: str) -> str:
    if not file_extension:
        raise ValueError("file_extension must be non-empty.")
    return file_extension if file_extension.startswith(".") else f".{file_extension}"

def _export_to_file(content: str, file_extension: str, directory: str = ".", prefix: str = "artifact") -> str:
    os.makedirs(directory, exist_ok=True)
    ext = _normalize_ext(file_extension)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(directory, f"{_safe_slug(prefix)}-{ts}{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def _call_llm(model: str, prompt: str, url: str = "http://localhost:11434/api/generate") -> str:
    """Stream response from an Ollama-compatible endpoint."""
    import requests
    data = {"model": model, "prompt": prompt}
    headers = {"Content-Type": "application/json"}
    full = []
    resp = requests.post(url, data=json.dumps(data), headers=headers, stream=True, timeout=600)
    try:
        for line in resp.iter_lines():
            if not line:
                continue
            d = json.loads(line.decode("utf-8"))
            chunk = d.get("response", "")
            if chunk:
                full.append(chunk)
    finally:
        resp.close()
    return "".join(full)

# =============================================================================
# Ecore / XMI parsing
# =============================================================================

ECORE_NS = "http://www.eclipse.org/emf/2002/Ecore"
XMI_NS   = "http://www.omg.org/XMI"

def _tag_local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag

def parse_ecore_schema(ecore_path: str) -> Dict:
    """
    Parse an .ecore file and extract:
      - package name, nsURI, nsPrefix
      - classes: name, supertypes, attributes (name, type, bounds), references (name, type, containment, bounds, eOpposite)
      - enums: name, literals
      - datatypes: name, instanceTypeName
    """
    tree = ET.parse(ecore_path)
    root = tree.getroot()

    if _tag_local(root.tag) not in ("EPackage", "package"):
        raise ValueError("Root element is not EPackage")

    pkg = {
        "name": root.attrib.get("name"),
        "nsURI": root.attrib.get("nsURI"),
        "nsPrefix": root.attrib.get("nsPrefix"),
        "classes": {},
        "enums": {},
        "datatypes": {}
    }

    for el in root.findall(".//*"):
        lname = _tag_local(el.tag)
        if lname == "EClass":
            name = el.attrib.get("name")
            supertypes = [st.split("#")[-1] for st in el.attrib.get("eSuperTypes", "").split(" ") if st]
            cls = {"name": name, "supertypes": supertypes, "attributes": [], "references": []}
            for feat in el:
                fl = _tag_local(feat.tag)
                if fl == "EAttribute":
                    cls["attributes"].append({
                        "name": feat.attrib.get("name"),
                        "type": feat.attrib.get("eType", "").split("#")[-1] or feat.attrib.get("type"),
                        "lower": int(feat.attrib.get("lowerBound", "0") or "0"),
                        "upper": feat.attrib.get("upperBound", "1") or "1",
                    })
                elif fl == "EReference":
                    cls["references"].append({
                        "name": feat.attrib.get("name"),
                        "type": feat.attrib.get("eType", "").split("#")[-1] or feat.attrib.get("type"),
                        "containment": feat.attrib.get("containment", "false") == "true",
                        "lower": int(feat.attrib.get("lowerBound", "0") or "0"),
                        "upper": feat.attrib.get("upperBound", "1") or "1",
                        "eOpposite": feat.attrib.get("eOpposite")
                    })
            pkg["classes"][name] = cls

        elif lname == "EEnum":
            name = el.attrib.get("name")
            literals = []
            for lit in el:
                if _tag_local(lit.tag) == "EEnumLiteral":
                    literals.append(lit.attrib.get("name"))
            pkg["enums"][name] = {"name": name, "literals": literals}

        elif lname == "EDataType":
            name = el.attrib.get("name")
            pkg["datatypes"][name] = {
                "name": name,
                "instanceTypeName": el.attrib.get("instanceTypeName")
            }

    return pkg

# =============================================================================
# Summaries & allow-list
# =============================================================================

def summarize_ecore_schema(schema: Dict, max_classes: int = 30, max_features: int = 12) -> str:
    lines = []
    lines.append(f"Package: {schema.get('name')} nsURI={schema.get('nsURI')} nsPrefix={schema.get('nsPrefix')}")
    if schema.get("enums"):
        for en, data in list(schema["enums"].items())[:10]:
            lits = ", ".join(data.get("literals", [])[:12])
            lines.append(f"Enum {en} :: {lits}")
    for i, (cn, cls) in enumerate(schema.get("classes", {}).items()):
        if i >= max_classes:
            lines.append(f"... (+{len(schema['classes']) - max_classes} more classes)")
            break
        sups = f" <: {', '.join(cls.get('supertypes', []) )}" if cls.get("supertypes") else ""
        lines.append(f"Class {cn}{sups}")
        for a in cls.get("attributes", [])[:max_features]:
            ub = a['upper']
            lines.append(f"  attr {a['name']} : {a['type']} [{a['lower']}..{ub}]")
        for r in cls.get("references", [])[:max_features]:
            ub = r['upper']
            c = " contains" if r.get("containment") else ""
            lines.append(f"  ref  {r['name']} -> {r['type']} [{r['lower']}..{ub}]{c}")
    return "\n".join(lines)

def list_allowed_identifiers(schema_domain: Dict, schema_rec: Optional[Dict]) -> List[str]:
    """
    Allow-list to reduce hallucinations; but the validator will enforce that *instances*
    must be of DOMAIN classes only.
    """
    ids = []
    for schema in (schema_domain, schema_rec or {}):
        if not schema:
            continue
        for cn, cls in schema.get("classes", {}).items():
            ids.append(cn)
            for a in cls.get("attributes", []):
                ids.append(a["name"])
            for r in cls.get("references", []):
                ids.append(r["name"])
        for en, data in schema.get("enums", {}).items():
            ids.append(en)
            ids.extend(data.get("literals", []))
        ids.extend(schema.get("datatypes", {}).keys())
    seen, out = set(), []
    for x in ids:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

# =============================================================================
# Few-shot loading (optional)
# =============================================================================

def _load_few_shots(paths: Optional[List[str]]) -> List[Tuple[str, str]]:
    shots = []
    if not paths:
        return shots
    for p in paths:
        if not p:
            continue
        name = os.path.basename(p)
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = f.read()
            shots.append((name, _escape_braces(raw)))
        except Exception as e:
            shots.append((f"{name} (unreadable)", _escape_braces(f"[ERROR READING FILE: {e}]")))
    return shots

# =============================================================================
# PROMPT (XMI-only, conforms to DOMAIN metamodel)
# =============================================================================

def build_xmi_artifact_prompt(
    domain_schema: Dict,
    recomm_schema: Optional[Dict] = None,
    domain_instances_summary: Optional[str] = None,
    recomm_instances_summary: Optional[str] = None,
    few_shot_paths: Optional[List[str]] = None,
    artifact_purpose: str = "Produce an XMI instance model that represents domain data and its interaction hooks with a recommender system, while STRICTLY conforming to the DOMAIN metamodel.",
) -> str:
    """
    Build a grounded prompt that forces STRICT XMI output *valid against the domain Ecore*.

    Model MUST output a single XMI document:

    <xmi:XMI xmlns:xmi="http://www.omg.org/XMI"
             xmlns:domain="<DOMAIN nsURI>">
      <!-- Instances must be elements in the 'domain' namespace, e.g., <domain:ClassName ...> -->
      <!-- Every instance must carry unique xmi:id; references must use xmi:idref OR nested elements per containment -->
    </xmi:XMI>

    HARD RULE: All created instances MUST be of DOMAIN EClasses (not recommender EClasses).
    If recommender-related info is needed, encode as attributes or references that exist in DOMAIN metamodel; otherwise use the literal "TODO_UNKNOWN".
    """
    domain_summary = summarize_ecore_schema(domain_schema)
    recomm_summary = summarize_ecore_schema(recomm_schema) if recomm_schema else "(not provided)"

    shots = _load_few_shots(few_shot_paths)
    examples_section = ""
    if shots:
        examples_section = "\nFEW-SHOT EXAMPLES (REFERENCE ONLY; DO NOT ECHO PROSE)\n" + "\n".join(
            f"--- EXAMPLE {i+1}: {name} ---\n{content}\n--- END EXAMPLE {i+1} ---"
            for i, (name, content) in enumerate(shots)
        )

    allowlist = list_allowed_identifiers(domain_schema, recomm_schema)
    allow_block = "ALLOWED IDENTIFIERS (prefer exact reuse; do not invent):\n" + \
                  ", ".join(_escape_braces(x) for x in allowlist[:300])

    anti_hallucination = f"""
ANTI-HALLUCINATION (STRICT)
- Create instances ONLY for DOMAIN EClasses from the domain package below.
- Attribute and reference names MUST exist in the DOMAIN metamodel; if missing, write "TODO_UNKNOWN".
- DO NOT invent classes, attributes, references, enum literals, or datatypes that are not present.
- All IDs must be unique (xmi:id), references must resolve (xmi:idref).
- Output ONE valid XML document only, no markdown fences, no commentary.
- Ensure a correct XMI root and namespaces:
  <xmi:XMI xmlns:xmi="http://www.omg.org/XMI" xmlns:domain="{_escape_braces(domain_schema.get('nsURI') or '')}">
""".strip()

    output_contract = """
OUTPUT SKELETON (EXAMPLE SHAPE; ADAPT TO YOUR INSTANCES)
<xmi:XMI xmlns:xmi="http://www.omg.org/XMI" xmlns:domain="DOMAIN_NSURI">
  <domain:ClassA xmi:id="a1" attr1="..." attr2="...">
    <!-- containment references can be nested here as <domain:ChildClass .../> -->
  </domain:ClassA>
  <domain:ClassB xmi:id="b1" refToA="a1"/>
</xmi:XMI>
""".strip()

    prompt = f"""
You are a modeling assistant that MUST produce an XMI instance model
STRICTLY conforming to the DOMAIN Ecore metamodel.

PURPOSE
- {_escape_braces(artifact_purpose)}

{anti_hallucination}

GROUNDING CHECKLIST
1) Only instantiate DOMAIN classes; verify every attribute/reference name exists in DOMAIN metamodel.
2) Use xmi:id for each instance; resolve references via xmi:idref or nested containment elements.
3) Respect multiplicities where possible; avoid introducing elements beyond the metamodel.
4) If something is missing in DOMAIN, use "TODO_UNKNOWN" instead of inventing new identifiers.
5) Ensure output is well-formed XML and valid XMI.

OUTPUT REQUIREMENTS
- Return ONE XMI document only (no extra text).
- Root element: <xmi:XMI ...>
- Namespaces: xmlns:xmi="http://www.omg.org/XMI", xmlns:domain="{_escape_braces(domain_schema.get('nsURI') or '')}"
- Instance elements MUST be in the 'domain' namespace: <domain:ClassName .../>

{output_contract}
{examples_section}

=== DOMAIN ECORE SUMMARY ===
{_escape_braces(domain_summary)}

=== RECOMMENDER ECORE SUMMARY (for context only; DO NOT instantiate these classes) ===
{_escape_braces(recomm_summary)}

{f"=== DOMAIN XMI SUMMARY (optional grounding) ===\n{_escape_braces(domain_instances_summary)}" if domain_instances_summary else ""}
{f"=== RECOMMENDER XMI SUMMARY (context only) ===\n{_escape_braces(recomm_instances_summary)}" if recomm_instances_summary else ""}

{allow_block}

REMINDER
- Output ONE valid XMI document only, with domain instances that conform to the DOMAIN metamodel.
""".strip()

    return prompt

# =============================================================================
# Generation
# =============================================================================

def generate_xmi_artifact_from_ecore(
    model: str,
    domain_ecore_path: str,
    recomm_ecore_path: Optional[str] = None,
    domain_xmi_path: Optional[str] = None,
    recomm_xmi_path: Optional[str] = None,
    few_shot_paths: Optional[List[str]] = None,
    export_directory: str = "./out_ecore_xmi",
    export_prefix: Optional[str] = None
) -> Tuple[str, str]:
    """
    Build a grounded XMI-only prompt and generate an XMI that conforms to the DOMAIN metamodel.
    Returns (payload, path).
    """
    # Parse schemas
    domain_schema = parse_ecore_schema(domain_ecore_path)
    recomm_schema = parse_ecore_schema(recomm_ecore_path) if recomm_ecore_path and os.path.exists(recomm_ecore_path) else None

    # Optional: instance summaries (kept short & optional, omitted here for brevity)
    dom_xmi_summary = None
    rec_xmi_summary = None

    # Prompt
    prompt = build_xmi_artifact_prompt(
        domain_schema=domain_schema,
        recomm_schema=recomm_schema,
        domain_instances_summary=dom_xmi_summary,
        recomm_instances_summary=rec_xmi_summary,
        few_shot_paths=few_shot_paths,
    )

    payload = _call_llm(model=model, prompt=prompt)

    if export_prefix is None:
        domain_stub = _safe_slug(os.path.splitext(os.path.basename(domain_ecore_path))[0])
        prefix = f"artifact-xmi-{domain_stub}"
    else:
        prefix = export_prefix

    out_path = _export_to_file(payload, file_extension=".xmi", directory=export_directory, prefix=prefix)
    return payload, out_path

# =============================================================================
# Validation — XMI vs DOMAIN Ecore (static checks, best-effort)
# =============================================================================

from typing import List, Dict, Tuple, Optional
import random
import xml.etree.ElementTree as ET

# Reuse constants from your module:
# XMI_NS = "http://www.omg.org/XMI"

# ------------------------------
# Tourism XMI generator (domain-aware)
# ------------------------------

def _pick_first_match(candidates: List[str], available: List[str]) -> Optional[str]:
    """Return the first candidate that exists in available (case-insensitive)."""
    lower = {a.lower(): a for a in available}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def _class_by_candidates(schema: Dict, candidates: List[str]) -> Optional[str]:
    """Pick a class name from schema['classes'] that matches any candidate."""
    if not schema or "classes" not in schema:
        return None
    names = list(schema["classes"].keys())
    return _pick_first_match(candidates, names)

def _attr_exists(schema: Dict, cls: str, attr: str) -> bool:
    try:
        return attr in {a["name"] for a in schema["classes"][cls]["attributes"]}
    except KeyError:
        return False

def _ref_exists(schema: Dict, cls: str, ref: str) -> bool:
    try:
        return ref in {r["name"] for r in schema["classes"][cls]["references"]}
    except KeyError:
        return False

def _best_attr(schema: Dict, cls: str, candidates: List[str]) -> Optional[str]:
    attrs = [a["name"] for a in schema["classes"].get(cls, {}).get("attributes", [])]
    return _pick_first_match(candidates, attrs)

def _best_ref(schema: Dict, cls: str, candidates: List[str]) -> Optional[str]:
    refs = [r["name"] for r in schema["classes"].get(cls, {}).get("references", [])]
    return _pick_first_match(candidates, refs)

def _ns_tag(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"

def generate_tourism_xmi_instances(
    domain_schema: Dict,
    num_users: int,
    num_ratings: int,
    preference_types: List[str],
    random_seed: int = 7,
    id_prefix: str = "t",
) -> str:
    """
    Create an XMI document with synthetic tourism data that CONFORMS (best-effort)
    to the DOMAIN metamodel.

    Parameters:
        domain_schema     : parsed Ecore schema (from parse_ecore_schema)
        num_users         : number of user/tourist instances to create
        num_ratings       : total number of rating instances to create
        preference_types  : e.g., ["Beach", "Museum", "Hiking", "Food"]
        random_seed       : determinism for reproducibility
        id_prefix         : prefix for xmi:id values

    Output:
        XMI string with <xmi:XMI ... xmlns:domain="nsURI"> and domain elements.
    """
    random.seed(random_seed)

    ns_uri = domain_schema.get("nsURI") or "TODO_UNKNOWN_NSURI"

    # --- Pick class names from the metamodel (heuristics) ---
    user_cls = _class_by_candidates(domain_schema, ["User", "Tourist", "Traveler", "Customer", "Guest", "Person"])
    item_cls = _class_by_candidates(domain_schema, ["Destination", "Attraction", "POI", "Hotel", "Place", "Activity", "Venue"])
    rating_cls = _class_by_candidates(domain_schema, ["Rating", "Review", "Score", "Feedback", "Evaluation"])
    pref_cls = _class_by_candidates(domain_schema, ["Preference", "Interest", "UserPreference", "CategoryPreference"])

    # Fallbacks (we keep names valid—even if they may not exist in metamodel; validator will flag it):
    user_cls = user_cls or "TODO_UNKNOWN_User"
    item_cls = item_cls or "TODO_UNKNOWN_Destination"
    rating_cls = rating_cls or "TODO_UNKNOWN_Rating"
    pref_cls = pref_cls or "TODO_UNKNOWN_Preference"

    # --- Attributes / references (best-effort mapping) ---
    # Common attribute guesses
    user_name_attr = _best_attr(domain_schema, user_cls, ["name", "fullName", "username", "label"])
    user_email_attr = _best_attr(domain_schema, user_cls, ["email", "mail"])
    user_age_attr = _best_attr(domain_schema, user_cls, ["age"])
    item_name_attr = _best_attr(domain_schema, item_cls, ["name", "title", "label"])
    item_type_attr = _best_attr(domain_schema, item_cls, ["type", "category", "kind"])
    rating_value_attr = _best_attr(domain_schema, rating_cls, ["value", "score", "stars", "ratingValue"])
    rating_ts_attr = _best_attr(domain_schema, rating_cls, ["timestamp", "time", "createdAt"])
    pref_type_attr = _best_attr(domain_schema, pref_cls, ["type", "category", "kind", "name"])
    pref_weight_attr = _best_attr(domain_schema, pref_cls, ["weight", "priority", "score", "strength"])

    # References
    r_user_ref = _best_ref(domain_schema, rating_cls, ["user", "tourist", "traveler", "author", "rater"])
    r_item_ref = _best_ref(domain_schema, rating_cls, ["item", "destination", "poi", "hotel", "place", "activity", "target"])
    u_prefs_ref = _best_ref(domain_schema, user_cls, ["preferences", "interests", "prefs"])
    u_ratings_ref = _best_ref(domain_schema, user_cls, ["ratings", "reviews"])

    # --- Build XML ---
    root = ET.Element(_ns_tag(XMI_NS, "XMI"))
    root.set("xmlns:xmi", XMI_NS)
    root.set("xmlns:domain", ns_uri)

    # Create some items (destinations/POIs) – ensure we have at least a few targets for ratings
    num_items = max(5, min(50, max(1, num_ratings // 2)))
    item_ids = []
    for i in range(1, num_items + 1):
        e = ET.SubElement(root, _ns_tag(ns_uri, item_cls))
        e.set(_ns_tag(XMI_NS, "id"), f"{id_prefix}-item-{i}")
        item_ids.append(f"{id_prefix}-item-{i}")
        # attributes
        e.set(item_name_attr or "name", f"Item-{i}" if item_name_attr else "TODO_UNKNOWN")
        if item_type_attr:
            e.set(item_type_attr, random.choice(["Beach", "City", "Mountain", "Museum", "Food", "Hiking"]))
        # else: omit

    # Users
    user_ids = []
    for i in range(1, num_users + 1):
        u = ET.SubElement(root, _ns_tag(ns_uri, user_cls))
        u.set(_ns_tag(XMI_NS, "id"), f"{id_prefix}-user-{i}")
        user_ids.append(f"{id_prefix}-user-{i}")
        if user_name_attr:
            u.set(user_name_attr, f"User {i}")
        if user_email_attr:
            u.set(user_email_attr, f"user{i}@example.org")
        if user_age_attr:
            u.set(user_age_attr, str(random.randint(18, 70)))
        # preferences as separate instances (linked to user via reference if model supports it)
        # We'll create one pref per listed type for each user (lightweight)
        if pref_cls:
            for ptype in preference_types:
                p = ET.SubElement(root, _ns_tag(ns_uri, pref_cls))
                p.set(_ns_tag(XMI_NS, "id"), f"{id_prefix}-pref-{i}-{_safe_slug(ptype)}")
                p.set(pref_type_attr or "type", ptype if pref_type_attr else "TODO_UNKNOWN")
                if pref_weight_attr:
                    p.set(pref_weight_attr, f"{random.randint(1, 5)}")
                # If the user class has a containment or attribute-like reference for preferences,
                # link it by setting the ref as idref attribute:
                if u_prefs_ref:
                    # attach on the user element
                    current = u.get(u_prefs_ref)
                    u.set(u_prefs_ref, ((current + " ") if current else "") + f"{id_prefix}-pref-{i}-{_safe_slug(ptype)}")

    # Ratings
    for r in range(1, num_ratings + 1):
        rid = f"{id_prefix}-rating-{r}"
        uref = random.choice(user_ids) if user_ids else None
        iref = random.choice(item_ids) if item_ids else None
        rt = ET.SubElement(root, _ns_tag(ns_uri, rating_cls))
        rt.set(_ns_tag(XMI_NS, "id"), rid)
        if rating_value_attr:
            rt.set(rating_value_attr, str(random.randint(1, 5)))
        if rating_ts_attr:
            rt.set(rating_ts_attr, f"2025-01-{random.randint(1,28):02d}T12:{random.randint(0,59):02d}:00Z")
        # references to user/item if available as attributes (idrefs)
        if r_user_ref and uref:
            rt.set(r_user_ref, uref)
        if r_item_ref and iref:
            rt.set(r_item_ref, iref)
        # also link ratings on user if a back-reference exists
        if u_ratings_ref and uref:
            # find the user element by id
            # (simple scan; small doc so it's fine)
            for el in root:
                if el.get(_ns_tag(XMI_NS, "id")) == uref:
                    cur = el.get(u_ratings_ref)
                    el.set(u_ratings_ref, ((cur + " ") if cur else "") + rid)
                    break

    # Serialize
    xml_bytes = ET.tostring(root, encoding="utf-8", method="xml")
    return xml_bytes.decode("utf-8")


def generate_and_validate_tourism_xmi(
    model: str,
    domain_ecore_path: str,
    num_users: int,
    num_ratings: int,
    preference_types: List[str],
    use_llm: bool = False,
    few_shot_paths: Optional[List[str]] = None,
    export_directory: str = "./out_ecore_xmi",
    export_prefix: Optional[str] = None,
) -> Tuple[str, str, Dict]:
    """
    Convenience wrapper:
    - If use_llm=False: produce a synthetic tourism XMI directly (fast, schema-aware).
    - If use_llm=True : build a strict XMI prompt from the Ecore and ask your LLM to generate (adds creativity).
    Then validates the result against the domain metamodel using your existing validator.

    Returns: (xmi_payload, out_path, validation_report)
    """
    os.makedirs(export_directory, exist_ok=True)
    domain_schema = parse_ecore_schema(domain_ecore_path)

    if use_llm:
        # Build a domain-grounded prompt; ask the model to generate the tourism instances
        purpose = (
            f"Produce an XMI instance model for a tourism scenario with about "
            f"{num_users} users and {num_ratings} ratings, and user preferences in "
            f"{', '.join(preference_types)}. Instances MUST conform to the DOMAIN metamodel."
        )
        prompt = build_xmi_artifact_prompt(
            domain_schema=domain_schema,
            recomm_schema=None,
            few_shot_paths=few_shot_paths,
            artifact_purpose=purpose,
        )
        xmi_payload = _call_llm(model=model, prompt=prompt)
        prefix = export_prefix or f"tourism-xmi-llm"
    else:
        xmi_payload = generate_tourism_xmi_instances(
            domain_schema=domain_schema,
            num_users=num_users,
            num_ratings=num_ratings,
            preference_types=preference_types,
            id_prefix="tour",
        )
        prefix = export_prefix or f"tourism-xmi-synth"

    out_path = _export_to_file(xmi_payload, file_extension=".xmi", directory=export_directory, prefix=prefix)

    # Validate
    report = validate_xmi_against_domain(xmi_payload, domain_schema)
    return xmi_payload, out_path, report







def _index_domain(schema: Dict) -> Dict[str, Dict]:
    """Index DOMAIN classes -> attributes/refs for quick validation."""
    idx = {}
    for cname, cls in schema.get("classes", {}).items():
        attrs = {a["name"]: a for a in cls.get("attributes", [])}
        refs  = {r["name"]: r for r in cls.get("references", [])}
        idx[cname] = {"attributes": attrs, "references": refs}
    return idx

def validate_xmi_against_domain(xmi_payload: str, domain_schema: Dict) -> Dict:
    """
    Validate basic conformance of an XMI payload against the DOMAIN metamodel:
      - Well-formed XML; root is xmi:XMI; has xmi namespace
      - Instance elements are domain:* (based on DOMAIN nsURI binding in document)
      - Element local names are DOMAIN EClass names
      - Attributes/refs exist in the corresponding EClass
      - xmi:id unique; xmi:idref resolve
    Returns: {"is_valid": bool, "syntactic": [...], "semantic": [...], "stats": {...}}
    """
    syntactic, semantic, stats = [], [], {}

    # Parse XML
    try:
        root = ET.fromstring(xmi_payload)
    except ET.ParseError as e:
        syntactic.append(f"XML not well-formed: {e}")
        return {"is_valid": False, "syntactic": syntactic, "semantic": semantic, "stats": {"parsed": 0}}

    stats["parsed"] = 1

    # Check root
    if _tag_local(root.tag) != "XMI" and not root.tag.endswith("}XMI"):
        syntactic.append("Root element is not <xmi:XMI>.")

    # Collect namespace bindings from attributes (ElementTree limitation)
    nsmap = {}
    for k, v in root.attrib.items():
        if k.startswith("xmlns:"):
            nsmap[k.split(":", 1)[1]] = v
        elif k == "xmlns":
            nsmap["default"] = v

    if "xmi" not in nsmap or nsmap["xmi"] != XMI_NS:
        syntactic.append('Missing or incorrect xmlns:xmi="http://www.omg.org/XMI".')

    # Find the DOMAIN prefix used in the doc for domain nsURI
    domain_ns = domain_schema.get("nsURI")
    domain_prefix = None
    for pfx, uri in nsmap.items():
        if uri == domain_ns:
            domain_prefix = pfx
            break
    if not domain_prefix:
        syntactic.append(f'DOMAIN nsURI "{domain_ns}" is not bound to any prefix in XMI (missing xmlns:domain?).')

    # Build domain index
    d_idx = _index_domain(domain_schema)
    domain_classes = set(d_idx.keys())

    # Gather ids and idrefs; walk elements
    all_ids = {}
    dup_ids = set()
    idrefs = []
    domain_instance_count = 0

    def _lname(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    for el in root.iter():
        # IDs
        xid = el.attrib.get(f"{{{XMI_NS}}}id") or el.attrib.get("xmi:id")
        if xid:
            if xid in all_ids:
                dup_ids.add(xid)
            else:
                all_ids[xid] = el

        # Only validate instance elements under domain prefix (e.g., <domain:ClassName>)
        if ":" in el.tag and domain_prefix and el.tag.startswith(f"{{"):
            # ElementTree uses Clark notation {uri}local
            # We only check local class name against domain classes if namespace == domain_ns
            uri = el.tag.split("}")[0].strip("{")
            if uri != domain_ns:
                continue
            local = _lname(el.tag)
            if local not in domain_classes:
                syntactic.append(f'Element <{domain_prefix}:{local}> is not a DOMAIN EClass.')
                continue
            domain_instance_count += 1

            # Validate attributes and refs
            cls_spec = d_idx.get(local, {})
            attr_spec = cls_spec.get("attributes", {})
            ref_spec  = cls_spec.get("references", {})

            # Check attributes (ignore xmi:*)
            for k, v in el.attrib.items():
                lk = _lname(k)
                if lk in ("id", "type"):  # xmi:id, xmi:type are handled separately
                    continue
                if k.startswith("{"+XMI_NS+"}"):
                    continue
                # If it matches a reference name, it might be an IDREF attribute (non-containment), OK
                if lk in ref_spec:
                    # capture potential idref
                    if v:
                        idrefs.append(v)
                    continue
                # Otherwise must be an attribute in the class
                if lk not in attr_spec:
                    syntactic.append(f'Attribute "{lk}" is not defined for DOMAIN class "{local}".')

            # Check child elements as possible containment references
            for ch in el:
                child_local = _lname(ch.tag)
                # child should be in the domain ns (containment) and name must be a reference that is containment=True
                # We allow direct nested <domain:ChildClass .../> (typical EMF) or <referenceName> wrapper styles,
                # but with ElementTree we mainly see the QName. We do a pragmatic check:
                if child_local in domain_classes:
                    # Nested instance: OK, but ensure a matching containment reference exists in parent (by type)
                    # (We can’t infer the exact ref name; heuristically accept and rely on post-tooling)
                    pass
                else:
                    # wrapper name must be a containment reference on parent
                    if child_local not in ref_spec or not ref_spec[child_local]["containment"]:
                        syntactic.append(
                            f'Child element "{child_local}" under <{domain_prefix}:{local}> is not a containment reference in DOMAIN.'
                        )

            # Multiplicities: best-effort heuristic is out-of-scope without full graph; skipped.

        # Collect idrefs from simple child text nodes named like "<refName>xid</refName>" (common EMF style)
        for ch in el:
            if ch.text and ch.text.strip():
                idrefs.append(ch.text.strip())

    stats["domain_instances"] = domain_instance_count

    if dup_ids:
        syntactic.append(f"Duplicate xmi:id detected: {', '.join(sorted(dup_ids))}")

    # Resolve idrefs
    unresolved = [r for r in idrefs if r and r not in all_ids]
    if unresolved:
        syntactic.append(f"Unresolved IDREF(s): {', '.join(sorted(set(unresolved)))}")

    # Semantic heuristics
    if domain_instance_count == 0:
        semantic.append("No DOMAIN instances found in XMI.")

    is_valid = (len(syntactic) == 0)
    return {"is_valid": is_valid, "syntactic": syntactic, "semantic": semantic, "stats": stats}





# Assumes these exist in your module:
# - XMI_NS, parse_ecore_schema, summarize_xmi_instances, parse_xmi_instances
# - build_xmi_artifact_prompt (TARGET = DOMAIN), validate_xmi_against_domain
# - _safe_slug, _export_to_file, _call_llm


# ---------- LLM prompt for an arbitrary TARGET schema (not only domain) ----------
def build_xmi_artifact_prompt_for(
    target_schema: Dict,
    context_schema: Optional[Dict] = None,
    few_shot_paths: Optional[List[str]] = None,
    artifact_purpose: str = "Produce an XMI instance model that STRICTLY conforms to the TARGET metamodel.",
    target_label: str = "TARGET",
    context_label: str = "CONTEXT",
) -> str:
    """
    Strict XMI prompt where the *target_schema* is the one to instantiate.
    The context_schema (if any) is only grounding; do NOT instantiate those classes.
    """
    # Reuse helpers from your module
    target_summary = summarize_ecore_schema(target_schema)
    context_summary = summarize_ecore_schema(context_schema) if context_schema else "(not provided)"

    # Few-shot examples
    shots = _load_few_shots(few_shot_paths)
    examples_section = ""
    if shots:
        examples_section = "\nFEW-SHOT EXAMPLES (REFERENCE ONLY; DO NOT ECHO PROSE)\n" + "\n".join(
            f"--- EXAMPLE {i+1}: {name} ---\n{content}\n--- END EXAMPLE {i+1} ---"
            for i, (name, content) in enumerate(shots)
        )

    nsuri = target_schema.get("nsURI") or ""
    anti_hallucination = f"""
ANTI-HALLUCINATION (STRICT)
- Instantiate ONLY {target_label} classes from the {target_label} metamodel below.
- Attribute/reference names MUST exist in the {target_label} metamodel; if missing, write "TODO_UNKNOWN".
- DO NOT invent classes/attributes/references/enums/datatypes that are not present.
- All IDs must be unique (xmi:id); references must resolve (xmi:idref) or be nested containment.
- Output ONE valid XML document only, no markdown fences, no commentary.
- Root element and namespaces:
  <xmi:XMI xmlns:xmi="http://www.omg.org/XMI" xmlns:{target_label.lower()}="{nsuri}">
""".strip()

    output_contract = f"""
OUTPUT SKELETON (SHAPE EXAMPLE)
<xmi:XMI xmlns:xmi="http://www.omg.org/XMI" xmlns:{target_label.lower()}="{nsuri}">
  <{target_label.lower()}:ClassA xmi:id="a1" attr1="..."/>
  <{target_label.lower()}:ClassB xmi:id="b1" refToA="a1"/>
</xmi:XMI>
""".strip()

    prompt = f"""
You are a modeling assistant that MUST produce an XMI instance model
STRICTLY conforming to the {target_label} Ecore metamodel.

PURPOSE
- {_escape_braces(artifact_purpose)}

{anti_hallucination}

GROUNDING CHECKLIST
1) Only instantiate {target_label} classes; verify attribute/reference names exist in {target_label}.
2) Use xmi:id for each instance; resolve references via xmi:idref or nested containment elements.
3) Respect multiplicities where possible; avoid introducing non-metamodel elements.
4) If something is missing, use "TODO_UNKNOWN" instead of inventing new identifiers.
5) Ensure output is well-formed XML and valid XMI.

OUTPUT REQUIREMENTS
- Return ONE XMI document only (no extra text).
- Root element: <xmi:XMI ...>
- Namespaces: xmlns:xmi="http://www.omg.org/XMI", xmlns:{target_label.lower()}="{_escape_braces(nsuri)}"
- Instance elements MUST be in the '{target_label.lower()}' namespace: <{target_label.lower()}:ClassName .../>

{output_contract}
{examples_section}

=== {target_label} ECORE SUMMARY ===
{_escape_braces(target_summary)}

=== {context_label} ECORE SUMMARY (context only; DO NOT instantiate these classes) ===
{_escape_braces(context_summary)}

REMINDER
- Output ONE valid XMI document only, with instances that conform to the {target_label} metamodel.
""".strip()


    return prompt

def parse_xmi_instances(xmi_path: str) -> Dict:
    """
    Parse a generic .xmi file into:
      - nsmap (prefix -> URI)
      - instances: list of {id, type, attrs: {k:v}, refs: {k:[ids/uris]}}
    This is best-effort and schema-agnostic.
    """
    tree = ET.parse(xmi_path)
    root = tree.getroot()

    # nsmap collection from attributes (ElementTree limitation)
    nsmap = {}
    for k, v in root.attrib.items():
        if k.startswith("xmlns:"):
            nsmap[k.split(":", 1)[1]] = v
        elif k == "xmlns":
            nsmap["default"] = v

    instances = []


# ---------- Synthetic RS generator (schema-aware, no LLM) ----------
def generate_recommender_xmi_instances(
    recomm_schema: Dict,
    num_users: int,
    num_ratings: int,
    preference_types: List[str],
    random_seed: int = 17,
    id_prefix: str = "rs",
) -> str:
    """
    Create a synthetic recommender-system XMI that conforms (best-effort) to the RS metamodel:
    typical classes like UserProfile, Item, Interaction/Rating, PreferenceVector, Feature.
    """
    random.seed(random_seed)
    ns_uri = recomm_schema.get("nsURI") or "TODO_UNKNOWN_NSURI"

    # Candidate class names (mapped to your schema names by best-match)
    user_cls   = _class_by_candidates(recomm_schema, ["UserProfile", "User", "Profile", "Account", "Member"]) or "TODO_UNKNOWN_UserProfile"
    item_cls   = _class_by_candidates(recomm_schema, ["Item", "Product", "Content", "Entity", "Asset"]) or "TODO_UNKNOWN_Item"
    inter_cls  = _class_by_candidates(recomm_schema, ["Interaction", "Rating", "Event", "Feedback", "Impression"]) or "TODO_UNKNOWN_Interaction"
    prefv_cls  = _class_by_candidates(recomm_schema, ["PreferenceVector", "Embedding", "Vector", "UserPreference"]) or "TODO_UNKNOWN_PrefVector"
    feat_cls   = _class_by_candidates(recomm_schema, ["Feature", "Attribute", "Signal"]) or "TODO_UNKNOWN_Feature"

    # Attributes / refs (best-effort)
    u_name_attr  = _best_attr(recomm_schema, user_cls, ["name", "username", "label"])
    u_extid_attr = _best_attr(recomm_schema, user_cls, ["externalId", "userId", "id"])
    i_name_attr  = _best_attr(recomm_schema, item_cls, ["name", "title", "label"])
    i_type_attr  = _best_attr(recomm_schema, item_cls, ["type", "category", "kind"])
    r_val_attr   = _best_attr(recomm_schema, inter_cls, ["value", "score", "rating"])
    r_ts_attr    = _best_attr(recomm_schema, inter_cls, ["timestamp", "time", "ts", "createdAt"])
    v_space_attr = _best_attr(recomm_schema, prefv_cls, ["space", "model", "dim"])
    v_vals_attr  = _best_attr(recomm_schema, prefv_cls, ["values", "vector", "embedding"])
    f_name_attr  = _best_attr(recomm_schema, feat_cls,  ["name", "key"])
    f_val_attr   = _best_attr(recomm_schema, feat_cls,  ["value"])

    r_user_ref   = _best_ref(recomm_schema, inter_cls, ["user", "profile", "actor"])
    r_item_ref   = _best_ref(recomm_schema, inter_cls, ["item", "target", "content"])
    u_vec_ref    = _best_ref(recomm_schema, user_cls,   ["vector", "preference", "embedding"])
    u_feat_ref   = _best_ref(recomm_schema, user_cls,   ["features"])
    i_feat_ref   = _best_ref(recomm_schema, item_cls,   ["features"])
    v_owner_ref  = _best_ref(recomm_schema, prefv_cls,  ["owner", "user"])  # vector -> user

    root = ET.Element(_ns_tag(XMI_NS, "XMI"))
    root.set("xmlns:xmi", XMI_NS)
    root.set("xmlns:rs", ns_uri)

    # Items
    num_items = max(5, min(60, max(1, num_ratings // 2)))
    item_ids = []
    for i in range(1, num_items + 1):
        e = ET.SubElement(root, _ns_tag(ns_uri, item_cls))
        eid = f"{id_prefix}-item-{i}"
        e.set(_ns_tag(XMI_NS, "id"), eid)
        item_ids.append(eid)
        e.set(i_name_attr or "name", f"RS-Item-{i}" if i_name_attr else "TODO_UNKNOWN")
        if i_type_attr:
            e.set(i_type_attr, random.choice(["Beach", "Museum", "Hiking", "Food", "City", "Mountain"]))

        # Optional item features
        if i_feat_ref and feat_cls:
            for k in random.sample(["popularity","trend","season","familyFriendly","priceTier"], k=2):
                f = ET.SubElement(root, _ns_tag(ns_uri, feat_cls))
                fid = f"{id_prefix}-ifeat-{i}-{k}"
                f.set(_ns_tag(XMI_NS, "id"), fid)
                if f_name_attr: f.set(f_name_attr, k)
                if f_val_attr:  f.set(f_val_attr, random.choice(["low","med","high"]))
                cur = e.get(i_feat_ref)
                e.set(i_feat_ref, ((cur + " ") if cur else "") + fid)

    # Users + vectors + features
    user_ids = []
    for i in range(1, num_users + 1):
        u = ET.SubElement(root, _ns_tag(ns_uri, user_cls))
        uid = f"{id_prefix}-user-{i}"
        u.set(_ns_tag(XMI_NS, "id"), uid)
        user_ids.append(uid)
        if u_name_attr:  u.set(u_name_attr, f"RSUser {i}")
        if u_extid_attr: u.set(u_extid_attr, f"u{i:04d}")

        # Preference vector
        if prefv_cls:
            v = ET.SubElement(root, _ns_tag(ns_uri, prefv_cls))
            vid = f"{id_prefix}-vec-{i}"
            v.set(_ns_tag(XMI_NS, "id"), vid)
            if v_space_attr: v.set(v_space_attr, "tourism-v1")
            if v_vals_attr:  v.set(v_vals_attr, ",".join(str(random.random())[:6] for _ in range(8)))
            if v_owner_ref:
                v.set(v_owner_ref, uid)
            elif u_vec_ref:
                cur = u.get(u_vec_ref)
                u.set(u_vec_ref, ((cur + " ") if cur else "") + vid)

        # Optional user features
        if u_feat_ref and feat_cls:
            for k in preference_types[:2]:
                f = ET.SubElement(root, _ns_tag(ns_uri, feat_cls))
                fid = f"{id_prefix}-ufeat-{i}-{_safe_slug(k)}"
                f.set(_ns_tag(XMI_NS, "id"), fid)
                if f_name_attr: f.set(f_name_attr, k)
                if f_val_attr:  f.set(f_val_attr, random.choice(["low","med","high"]))
                cur = u.get(u_feat_ref)
                u.set(u_feat_ref, ((cur + " ") if cur else "") + fid)

    # Interactions / Ratings
    for r in range(1, num_ratings + 1):
        rid = f"{id_prefix}-inter-{r}"
        uref = random.choice(user_ids) if user_ids else None
        iref = random.choice(item_ids) if item_ids else None

        rt = ET.SubElement(root, _ns_tag(ns_uri, inter_cls))
        rt.set(_ns_tag(XMI_NS, "id"), rid)
        if r_val_attr: rt.set(r_val_attr, str(random.randint(1, 5)))
        if r_ts_attr:  rt.set(r_ts_attr, f"2025-02-{random.randint(1,28):02d}T12:{random.randint(0,59):02d}:00Z")
        if r_user_ref and uref: rt.set(r_user_ref, uref)
        if r_item_ref and iref: rt.set(r_item_ref, iref)

    return ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")


def summarize_xmi_instances(xmi: Dict, max_instances: int = 50, max_fields: int = 8) -> str:
    lines = [f"XMI nsmap: {xmi.get('nsmap', {})}"]
    for i, inst in enumerate(xmi.get("instances", [])):
        if i >= max_instances:
            lines.append(f"... (+{len(xmi['instances']) - max_instances} more instances)")
            break
        head = f"{inst.get('type','?')}#{inst.get('id','?')}"
        lines.append(f"- {head}")
        # attrs
        for j, (k, v) in enumerate(inst.get("attrs", {}).items()):
            if j >= max_fields:
                lines.append("  ...")
                break
            lines.append(f"  @ {k} = {v}")
        # refs
        for j, (k, arr) in enumerate(inst.get("refs', {}").items()):
            if j >= max_fields:
                lines.append("  ...")
                break
            lines.append(f"  -> {k} : {arr}")
    return "\n".join(lines)

# ---------- Orchestrator: produce BOTH XMI files ----------
def generate_and_validate_tourism_and_rs_xmi(
    model: str,
    domain_ecore_path: str,
    recomm_ecore_path: str,
    num_users: int,
    num_ratings: int,
    preference_types: List[str],
    *,
    # Few-shots for each side (can reuse same list)
    few_shot_paths_domain: Optional[List[str]] = None,
    few_shot_paths_recomm: Optional[List[str]] = None,
    # LLM or synthetic per side
    use_llm_domain: bool = False,
    use_llm_recomm: bool = False,
    # Optional RS seed XMI for grounding in prompt (context only)
    recomm_xmi_path: Optional[str] = None,

    export_directory: str = "./out_ecore_xmi",
    export_prefix_domain: Optional[str] = None,
    export_prefix_recomm: Optional[str] = None,
) -> Tuple[str, str, Dict, str, str, Dict]:
    """
    Generate TWO XMI outputs:
      1) DOMAIN tourism XMI (conforms to DOMAIN Ecore)
      2) RECOMMENDER-SYSTEM XMI (conforms to RS Ecore)

    Returns:
      (domain_payload, domain_path, domain_report, recomm_payload, recomm_path, recomm_report)
    """
    os.makedirs(export_directory, exist_ok=True)

    domain_schema = parse_ecore_schema(domain_ecore_path)
    recomm_schema = parse_ecore_schema(recomm_ecore_path)

    # ---- DOMAIN XMI ----
    if use_llm_domain:
        purpose_dom = (
            f"Tourism scenario: ~{num_users} users and ~{num_ratings} ratings; "
            f"preferences = {', '.join(preference_types)}. "
            "Instantiate DOMAIN classes only."
        )
        # reuse your domain-specific prompt builder
        prompt_dom = build_xmi_artifact_prompt(
            domain_schema=domain_schema,
            recomm_schema=recomm_schema,  # context allowed
            domain_instances_summary=None,
            recomm_instances_summary=(
                summarize_xmi_instances(parse_xmi_instances(recomm_xmi_path))
                if recomm_xmi_path and os.path.exists(recomm_xmi_path) else None
            ),
            few_shot_paths=few_shot_paths_domain,
            artifact_purpose=purpose_dom,
        )
        domain_payload = _call_llm(model=model, prompt=prompt_dom)
        dom_prefix = export_prefix_domain or f"tourism-domain-llm"
    else:
        # use your existing synthetic generator for DOMAIN (already provided earlier)
        domain_payload = generate_tourism_xmi_instances(
            domain_schema=domain_schema,
            num_users=num_users,
            num_ratings=num_ratings,
            preference_types=preference_types,
            id_prefix="tour",
        )
        dom_prefix = export_prefix_domain or f"tourism-domain-synth"

    domain_path = _export_to_file(domain_payload, file_extension=".xmi", directory=export_directory, prefix=dom_prefix)
    domain_report = validate_xmi_against_domain(domain_payload, domain_schema)

    # ---- RECOMMENDER XMI ----
    if use_llm_recomm:
        purpose_rs = (
            f"Recommender-system scenario consistent with tourism domain scale: "
            f"~{num_users} users, ~{num_ratings} interactions; "
            f"preference types = {', '.join(preference_types)}. "
            "Instantiate RECOMMENDER-SYSTEM (TARGET) classes only."
        )
        prompt_rs = build_xmi_artifact_prompt_for(
            target_schema=recomm_schema,
            context_schema=domain_schema,         # domain as CONTEXT
            few_shot_paths=few_shot_paths_recomm,
            artifact_purpose=purpose_rs,
            target_label="RS",
            context_label="DOMAIN",
        )
        recomm_payload = _call_llm(model=model, prompt=prompt_rs)
        rs_prefix = export_prefix_recomm or f"tourism-recomm-llm"
    else:
        recomm_payload = generate_recommender_xmi_instances(
            recomm_schema=recomm_schema,
            num_users=num_users,
            num_ratings=num_ratings,
            preference_types=preference_types,
            id_prefix="rs",
        )
        rs_prefix = export_prefix_recomm or f"tourism-recomm-synth"

    recomm_path = _export_to_file(recomm_payload, file_extension=".xmi", directory=export_directory, prefix=rs_prefix)

    # Validate RS XMI against the RS schema (reuse same validator; it's schema-agnostic except name)
    recomm_report = validate_xmi_against_domain(recomm_payload, recomm_schema)

    return domain_payload, domain_path, domain_report, recomm_payload, recomm_path, recomm_report


# =============================================================================
# Example MAIN (direct inputs; no assignments)
# =============================================================================

if __name__ == "__main__":
    MODEL = "gpt-oss:120b-cloud"
    DOMAIN_ECORE = "./input_models/large/domain.ecore"
    RS_ECORE     = "./input_models/large/recommendersystemGeneric.ecore"
    SHOTS_DOMAIN = ["./input_models/large/domain.model"]   # optional
    SHOTS_RS     = ["./input_models/large/recommendersystemGeneric.model"]    # optional

    (dom_xmi, dom_path, dom_rep,
     rs_xmi,  rs_path,  rs_rep) = generate_and_validate_tourism_and_rs_xmi(
        model=MODEL,
        domain_ecore_path=DOMAIN_ECORE,
        recomm_ecore_path=RS_ECORE,
        num_users=10,
        num_ratings=20,
        preference_types=["Beach", "Museum", "Hiking", "Food"],
        few_shot_paths_domain=SHOTS_DOMAIN,
        few_shot_paths_recomm=SHOTS_RS,
        use_llm_domain=True,     # LLM for DOMAIN?
        use_llm_recomm=False,    # synthetic for RS?
        export_directory="./out_dual_xmi"
    )

    print("DOMAIN XMI:", dom_path, "valid:", dom_rep["is_valid"])
    if dom_rep["syntactic"]: print("  domain syntactic:", dom_rep["syntactic"])
    if dom_rep["semantic"]:  print("  domain semantic :", dom_rep["semantic"])

    print("RS XMI:", rs_path, "valid:", rs_rep["is_valid"])
    if rs_rep["syntactic"]: print("  rs syntactic:", rs_rep["syntactic"])
    if rs_rep["semantic"]:  print("  rs semantic :", rs_rep["semantic"])