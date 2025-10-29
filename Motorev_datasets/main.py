# =========================
# MAIN XMI/.model PIPELINE
# =========================
from __future__ import annotations
import os, time, json
from typing import List, Optional, Tuple

from prompt_pipeline import _escape_braces, _call_llm, _export_to_file, _safe_slug
from xmi_model_adapter import parse_xmi_instances,summarize_xmi_instances, validate_cross_resource_hrefs

# ---------- Few-shot helpers ----------
def _read_few_shots(files: Optional[List[str]] = None) -> str:
    if not files:
        return ""
    shots = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                shots.append(f"--- FEW-SHOT FROM {os.path.basename(fp)} ---\n{f.read().strip()}")
        except Exception as e:
            shots.append(f"--- FEW-SHOT LOAD ERROR {os.path.basename(fp)} ---\n{e}")
    return "\n\n".join(shots)

# ---------- Prompt builders (XMI only) ----------
def _prompt_domain_xmi(
    domain_name: str,
    domain_description: str,
    few_shots_text: str = "",
) -> str:
    dn = _escape_braces(domain_name)
    dd = _escape_braces(domain_description)
    fs = _escape_braces(few_shots_text)

    return f"""
You are an EMF/XMI modeling assistant.

OBJECTIVE
- Generate a valid EMF XMI model (file extension .model) for the given DOMAIN.
- Use correct namespaces and xmi:version="2.0".
- Output ONLY the model payload (no prose, no markdown fences).
- Avoid hallucinations by using only entities/attributes derivable from the description and shots.

CONFORMANCE
- The result MUST be a well-formed XMI instance conforming to the domain metamodel "http://org.rs.domain".
- Provide stable xmi:id for all instances; use meaningful attribute values.
- Use xsi:type where subclassing applies (e.g., Indoor/Outdoor POIs).
- If any data seems ambiguous, omit it rather than inventing.

CROSS-FILE READINESS
- Ensure that elements that may be referenced by another file (e.g., recommender.model) have stable xmi:id values.

NAMESPACES
- xmlns:xmi="http://www.omg.org/XMI"
- xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
- xmlns:domain="http://org.rs.domain"

FEW-SHOTS (optional, for structure guidance only)
{fs}

DOMAIN NAME
{dn}

DOMAIN DESCRIPTION
{dd}

REMINDER
- Output XMI ONLY (no explanations).
""".strip()

def _prompt_rs_xmi(
    algorithm_name: str,
    rs_description: str,
    domain_model_filename: str,
    few_shots_text: str = "",
) -> str:
    rn = _escape_braces(algorithm_name)
    rd = _escape_braces(rs_description)
    fs = _escape_braces(few_shots_text)
    # NOTE: domain_model_filename should be a sibling like "domain.model" for hrefs: domain.model#_someId
    return f"""
You are an EMF/XMI modeling assistant.

OBJECTIVE
- Generate a valid EMF XMI model (file extension .model) for a Recommender System configuration named "{rn}".
- Use correct namespaces and xmi:version="2.0".
- Output ONLY the model payload (no prose, no markdown fences).
- Avoid hallucinations; use only structures derivable from description and few-shot patterns.

CONFORMANCE
- Conform to the recommender system metamodel "http://org.rs".
- Use xsi:type for concrete components (e.g., org.rs:HybridBased, CF/CB components).
- Data rows for ratings MUST link to DOMAIN model elements via cross-resource hrefs "domain.model#<xmi:id>".

CROSS-RESOURCE LINKS
- Any reference to users/items MUST employ href="{domain_model_filename}#<xmi:id>".
- Do not fabricate ids. If user/item not present in the domain, omit it rather than inventing.

NAMESPACES
- xmlns:xmi="http://www.omg.org/XMI"
- xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
- xmlns:org.rs="http://org.rs"
- xmlns:domain="http://org.rs.domain"

FEW-SHOTS (optional)
{fs}

RECOMMENDER DESCRIPTION
{rd}

REMINDER
- Output XMI ONLY (no explanations).
""".strip()

# ---------- Generators ----------
def generate_domain_model(
    model: str,
    domain_name: str,
    domain_description: str,
    export_dir: str,
    export_prefix: str = "domain",
    few_shot_files: Optional[List[str]] = None,
) -> Tuple[str, str]:
    shots = _read_few_shots(few_shot_files)
    prompt = _prompt_domain_xmi(domain_name=domain_name, domain_description=domain_description, few_shots_text=shots)
    payload = _call_llm(model=model, prompt=prompt)
    path = _export_to_file(payload, file_extension="model", directory=export_dir, prefix=export_prefix)
    return payload, path

def generate_recommender_model(
    model: str,
    algorithm_name: str,
    rs_description: str,
    export_dir: str,
    export_prefix: str = "recommender",
    domain_model_filename: str = "domain.model",
    few_shot_files: Optional[List[str]] = None,
) -> Tuple[str, str]:
    shots = _read_few_shots(few_shot_files)
    prompt = _prompt_rs_xmi(
        algorithm_name=algorithm_name,
        rs_description=rs_description,
        domain_model_filename=domain_model_filename,
        few_shots_text=shots,
    )
    payload = _call_llm(model=model, prompt=prompt)
    path = _export_to_file(payload, file_extension="model", directory=export_dir, prefix=export_prefix)
    return payload, path

# ---------- Validation wrappers (adapter-based) ----------
def validate_model_file(path: str) -> str:
    parsed = parse_xmi_instances(path)
    return summarize_xmi_instances(parsed)

def validate_cross_refs(recommender_path: str) -> dict:
    # Return full href report for programmatic decisions
    return validate_cross_resource_hrefs(recommender_path)

def compose_tourism_domain_description(
    num_users: int,
    preference_types: List[str],
) -> str:
    prefs_csv = ", ".join(preference_types) if preference_types else "priceRange, transportationMode, hikingSkill"
    return (
        "Tourism domain with entities: tourists (with profiles and preferences), and POIs (Indoor/Outdoor) with categories "
        "(INFOPOINT, MUSEO, AREA_FAUNISTICA, COMUNE).\n\n"
        f"REQUIREMENTS\n"
        f"- Create EXACTLY {num_users} tourist instances with stable xmi:id using pattern _gen_tourist_<n>.\n"
        f"- Each tourist MUST include a preference block with attributes drawn from: {prefs_csv}.\n"
        f"- Include a realistic but small set of POIs (mixed Indoor/Outdoor) and categories; reuse categories via xmi:id.\n"
        f"- Ensure all xmi:id values are unique and stable; no placeholders.\n"
        f"- Use namespace http://org.rs.domain; top-level root is domain:RSDomain; include name for the park.\n"
        f"- Avoid hallucinated attributes; if unsure, omit optional fields."
    )

def compose_tourism_recommender_description(
    num_ratings: int,
    preference_types: List[str],
) -> str:
    prefs_csv = ", ".join(preference_types) if preference_types else "priceRange, transportationMode, hikingSkill"
    return (
        "Hybrid tourism recommender (org.rs:HybridBased) combining collaborative and content-based signals.\n\n"
        "STRUCTURE\n"
        "- Root element: org.rs:Algorithm with name='TRS'.\n"
        "- Provide a <filteringRS xsi:type='org.rs:HybridBased'> that contains:\n"
        "  * A CF component with a <data> section containing rating rows.\n"
        "  * Optionally a CB component referencing content features aligned to domain preferences.\n\n"
        f"RATINGS\n"
        f"- Generate AT LEAST {num_ratings} rating rows as <rows value='float'>.\n"
        "- Each row MUST link a user and an item using cross-resource hrefs to domain.model (e.g., domain.model#_gen_tourist_21 and a POI id).\n"
        "- Distribute ratings across multiple users and items; values in [3.5, 5.0].\n\n"
        "CONSTRAINTS\n"
        "- All href targets MUST exist in domain.model.\n"
        f"- If using CB features, align them to the domain preference dimensions: {prefs_csv}.\n"
        "- Use proper namespaces: http://org.rs and http://org.rs.domain; xmi:version='2.0'.\n"
        "- No fabricated ids; if a target is missing, omit the row."
    )

# ---- Few-shot loader reused ----
def _read_few_shots(files: Optional[List[str]] = None) -> str:
    if not files: return ""
    blocks = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                blocks.append(f"--- FEW-SHOT FROM {os.path.basename(fp)} ---\n{f.read().strip()}")
        except Exception as e:
            blocks.append(f"--- FEW-SHOT LOAD ERROR {os.path.basename(fp)} ---\n{e}")
    return "\n\n".join(blocks)

# ---- Wire the tourism parameters into the existing generators ----
def main_xmi_pipeline_tourism():
    MODEL = "gpt-oss:120b-cloud"

    # Parameters requested (kept from your previous main style)
    NUM_USERS = 10                 # number of tourist instances to generate
    NUM_RATINGS = 20              # number of rating rows in RS
    PREFERENCE_TYPES = [           # preference dimensions to include
        "preferredPriceRange", "preferredTransportationMode", "hikingSkill"
    ]

    # Output folders
    out_root = "./out_models_old"
    os.makedirs(out_root, exist_ok=True)
    model_slug = _safe_slug(MODEL) or "model"
    export_dir = os.path.join(out_root, model_slug)
    os.makedirs(export_dir, exist_ok=True)

    # Optional few-shots
    few_shots = [
        "./input_models/large/domain.model",
        "./input_models/large/recommendersystemGeneric.model",
    ]

    # Build descriptions from parameters
    domain_name = "National Park of Abruzzo"
    domain_desc = compose_tourism_domain_description(
        num_users=NUM_USERS,
        preference_types=PREFERENCE_TYPES,
    )
    rs_desc = compose_tourism_recommender_description(
        num_ratings=NUM_RATINGS,
        preference_types=PREFERENCE_TYPES,
    )

    # ---- Generate DOMAIN ----
    t0 = time.perf_counter()
    domain_payload, domain_path = generate_domain_model(
        model=MODEL,
        domain_name=domain_name,
        domain_description=domain_desc + "\n\nFEW-SHOTS\n" + _read_few_shots(few_shots),
        export_dir=export_dir,
        export_prefix="domain",
        few_shot_files=None,  # already appended text above
    )
    t1 = time.perf_counter()
    print(f"[DOMAIN] Saved: {domain_path} ({t1 - t0:.2f}s)")
    print(validate_model_file(domain_path))

    # ---- Generate RS (hrefs pointing to sibling 'domain.model') ----
    t2 = time.perf_counter()
    rs_payload, rs_path = generate_recommender_model(
        model=MODEL,
        algorithm_name="TRS",
        rs_description=rs_desc + "\n\nFEW-SHOTS\n" + _read_few_shots(few_shots),
        export_dir=export_dir,
        export_prefix="recommender",
        domain_model_filename="domain.model",
        few_shot_files=None,  # already appended text above
    )
    t3 = time.perf_counter()
    print(f"[RS] Saved: {rs_path} ({t3 - t2:.2f}s)")
    print(validate_model_file(rs_path))

    # ---- Cross-resource validation (RS -> domain.model) ----
    href_report = validate_cross_refs(rs_path)
    ok_n = len(href_report.get("ok", []))
    miss_n = len(href_report.get("missing", []))
    print(f"[RS] Cross-resource hrefs: OK={ok_n}, Missing={miss_n}")
    if miss_n:
        print("Missing examples:", href_report["missing"][:10])



def main_xmi_pipeline_tourism_versions(
    N_VERSIONS: int = 5,
    MODEL: str = "gpt-oss:120b-cloud",
    NUM_USERS: int = 50,
    NUM_RATINGS: int = 600,
    PREFERENCE_TYPES: Optional[List[str]] = None,
    FEW_SHOTS: Optional[List[str]] = None,
    OUT_ROOT: str = "./out_models_old"
):
    """
    Generate N versions of:
      - domain.model  (tourism domain)
      - recommender.model (hybrid RS referencing domain.model)
    Validate each and print a compact summary.

    Assumes availability of:
      - _safe_slug, _read_few_shots
      - compose_tourism_domain_description, compose_tourism_recommender_description
      - generate_domain_model, generate_recommender_model
      - validate_model_file, validate_cross_refs
    """
    PREFERENCE_TYPES = PREFERENCE_TYPES or [
        "preferredPriceRange", "preferredTransportationMode", "hikingSkill"
    ]
    FEW_SHOTS = FEW_SHOTS or []  # paths to .model few-shot examples

    # Folder: ./out_models_old/<model_slug>/
    os.makedirs(OUT_ROOT, exist_ok=True)
    model_slug = _safe_slug(MODEL) or "model"
    export_root = os.path.join(OUT_ROOT, model_slug)
    os.makedirs(export_root, exist_ok=True)

    # Compose base descriptions (parameterized)
    domain_name = "National Park of Abruzzo"
    base_domain_desc = compose_tourism_domain_description(
        num_users=NUM_USERS, preference_types=PREFERENCE_TYPES
    )
    base_rs_desc = compose_tourism_recommender_description(
        num_ratings=NUM_RATINGS, preference_types=PREFERENCE_TYPES
    )
    few_shots_text = _read_few_shots(FEW_SHOTS)

    results = []

    for v in range(1, N_VERSIONS + 1):
        print(f"\n=== Version {v}/{N_VERSIONS} ===")

        # Optional per-version subfolder: ./out_models_old/<model_slug>/v{v}
        version_dir = os.path.join(export_root, f"v{v}")
        os.makedirs(version_dir, exist_ok=True)

        # ----- DOMAIN -----
        t0 = time.perf_counter()
        domain_payload, domain_path = generate_domain_model(
            model=MODEL,
            domain_name=domain_name,
            domain_description=f"{base_domain_desc}\n\nFEW-SHOTS\n{few_shots_text}",
            export_dir=version_dir,
            export_prefix="domain",
            few_shot_files=None,  # we already appended shots text
        )
        t1 = time.perf_counter()
        domain_secs = t1 - t0
        print(f"[DOMAIN] Saved: {domain_path} ({domain_secs:.2f}s)")
        domain_validation = validate_model_file(domain_path)

        # ----- RECOMMENDER (hrefs -> domain.model) -----
        t2 = time.perf_counter()
        rs_payload, rs_path = generate_recommender_model(
            model=MODEL,
            algorithm_name="TRS",
            rs_description=f"{base_rs_desc}\n\nFEW-SHOTS\n{few_shots_text}",
            export_dir=version_dir,
            export_prefix="recommender",
            domain_model_filename="domain.model",  # sibling file name for hrefs
            few_shot_files=None,                   # we already appended shots text
        )
        t3 = time.perf_counter()
        rs_secs = t3 - t2
        print(f"[RS] Saved: {rs_path} ({rs_secs:.2f}s)")
        rs_validation = validate_model_file(rs_path)

        # ----- Cross-resource validation -----
        href_report = validate_cross_refs(rs_path)
        ok_n = len(href_report.get("ok", []))
        miss_n = len(href_report.get("missing", []))
        print(f"[RS] Cross-resource hrefs: OK={ok_n}, Missing={miss_n}")
        if miss_n:
            print("Missing examples:", href_report["missing"][:10])

        # Collect run summary
        results.append({
            "version": v,
            "model": MODEL,
            "export_dir": version_dir,
            "domain_path": domain_path,
            "domain_time_s": round(domain_secs, 3),
            "domain_validation": domain_validation,
            "rs_path": rs_path,
            "rs_time_s": round(rs_secs, 3),
            "rs_validation": rs_validation,
            "href_ok": ok_n,
            "href_missing": miss_n,
        })

    # ----- Summary -----
    print("\n=== Summary ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    #main_xmi_pipeline_tourism()

    main_xmi_pipeline_tourism_versions(
        N_VERSIONS=5,
        MODEL="gpt-oss:120b-cloud",
        NUM_USERS=200,
        NUM_RATINGS=400,
        PREFERENCE_TYPES=[
            "preferredPriceRange", "preferredTransportationMode",
            "hikingSkill"
        ],
        FEW_SHOTS=[
            "./input_models/large/domain.model",
            "./input_models/large/recommendersystemGeneric.model",
        ],
        OUT_ROOT="./out_models_hybrid/large"
    )

