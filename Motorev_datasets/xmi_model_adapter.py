# =========================
# XMI / .model ADAPTER PATCH
# =========================
from __future__ import annotations
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple, List

# Keep your existing constant if present
XMI_NS = "http://www.omg.org/XMI"

# --- Helper: identify model-like files ---
def is_xmi_or_model(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".xmi", ".model"}

def _abs_join(base_dir: str, href_file: str) -> str:
    if os.path.isabs(href_file):
        return href_file
    return os.path.normpath(os.path.join(base_dir, href_file))

_href_rx = re.compile(r"(?P<file>[^#]+)#(?P<frag>.+)")

# --- Cache for loaded documents so cross-resource lookups are cheap ---
class _DocCache:
    def __init__(self):
        self._trees: Dict[str, ET.ElementTree] = {}
        self._id_index: Dict[str, Dict[str, ET.Element]] = {}

    def get_tree(self, path: str) -> Optional[ET.ElementTree]:
        return self._trees.get(os.path.abspath(path))

    def put_tree(self, path: str, tree: ET.ElementTree):
        ap = os.path.abspath(path)
        self._trees[ap] = tree

    def get_index(self, path: str) -> Optional[Dict[str, ET.Element]]:
        return self._id_index.get(os.path.abspath(path))

    def put_index(self, path: str, idx: Dict[str, ET.Element]):
        ap = os.path.abspath(path)
        self._id_index[ap] = idx

# --- Load any *.xmi or *.model as XML ---
def load_xmi_or_model(path: str, cache: Optional[_DocCache] = None) -> ET.ElementTree:
    if not is_xmi_or_model(path):
        raise ValueError(f"Unsupported file extension: {path}")
    ap = os.path.abspath(path)
    if cache:
        t = cache.get_tree(ap)
        if t is not None:
            return t
    tree = ET.parse(ap)
    if cache:
        cache.put_tree(ap, tree)
    return tree

# --- Build an xmi:id index for quick lookup ---
def index_xmi_ids(tree: ET.ElementTree, cache: Optional[_DocCache] = None, source_path: Optional[str] = None) -> Dict[str, ET.Element]:
    idx: Dict[str, ET.Element] = {}
    root = tree.getroot()
    xmi_id_attr = f"{{{XMI_NS}}}id"
    for el in root.iter():
        xid = el.attrib.get(xmi_id_attr)
        if xid:
            idx[xid] = el
    if cache and source_path:
        cache.put_index(source_path, idx)
    return idx

# --- Resolve a single EMF href like "domain.model#_abc123" ---
def parse_emf_href(href: str) -> Optional[Tuple[str, str]]:
    m = _href_rx.match(href.strip())
    if not m:
        return None
    return m.group("file"), m.group("frag")

def resolve_href(
    href: str,
    base_file: str,
    cache: Optional[_DocCache] = None
) -> Optional[ET.Element]:
    """
    Resolve an EMF-style cross-resource href.
    Returns the referenced element or None if not found.
    """
    parsed = parse_emf_href(href)
    if not parsed:
        return None
    href_file, frag = parsed
    base_dir = os.path.dirname(os.path.abspath(base_file))
    target_path = _abs_join(base_dir, href_file)
    # Load target document
    tcache = cache or _DocCache()
    tree = load_xmi_or_model(target_path, tcache)
    # Build or fetch id index
    idx = tcache.get_index(target_path)
    if idx is None:
        idx = index_xmi_ids(tree, tcache, target_path)
    # EMF fragments typically equal xmi:id
    return idx.get(frag)

# --- Validate cross-resource hrefs in a document (syntactic existence check only) ---
def validate_cross_resource_hrefs(
    path: str,
    cache: Optional[_DocCache] = None
) -> Dict[str, List[str]]:
    """
    Scans for attributes named 'href' and checks that referenced xmi:id exists
    in the target .xmi/.model.
    Returns dict with 'missing' and 'ok' lists of hrefs.
    """
    tcache = cache or _DocCache()
    tree = load_xmi_or_model(path, tcache)
    ok: List[str] = []
    missing: List[str] = []

    for el in tree.getroot().iter():
        # EMF uses 'href' (no namespace) for cross-resource proxies
        href = el.attrib.get("href")
        if href:
            resolved = resolve_href(href, path, tcache)
            if resolved is not None:
                ok.append(href)
            else:
                missing.append(href)

    return {"ok": ok, "missing": missing}

# --- Public: parse instances (backward compatible), now supports .model ---
def parse_xmi_instances(path: str) -> Dict:
    """
    Loads an XMI or .model file and produces a lightweight summary structure.
    (Kept compatible with the rest of your pipeline.)
    """
    cache = _DocCache()
    tree = load_xmi_or_model(path, cache)
    root = tree.getroot()

    # Namespace map (ElementTree doesn’t keep prefix map; we gather from attributes)
    nsmap = _collect_nsmap(root)

    # Basic listing of elements and ids
    xmi_id_attr = f"{{{XMI_NS}}}id"
    elements = []
    for el in root.iter():
        tag = _local_name(el.tag)
        ns = _ns_uri(el.tag)
        elements.append({
            "tag": tag,
            "ns": ns,
            "xmi_id": el.attrib.get(xmi_id_attr),
            "attrs": dict(el.attrib),
        })

    # Cross-resource check (non-fatal)
    href_report = validate_cross_resource_hrefs(path, cache)

    return {
        "path": os.path.abspath(path),
        "namespaces": nsmap,
        "count": len(elements),
        "elements": elements,
        "href_report": href_report,
    }

# --- Public: summarizer (unchanged API), enriched with href stats for .model ---
def summarize_xmi_instances(parsed: Dict) -> str:
    path = parsed.get("path", "")
    cnt = parsed.get("count", 0)
    ns = parsed.get("namespaces", {})
    href_rep = parsed.get("href_report", {})
    ok_n = len(href_rep.get("ok", []))
    miss_n = len(href_rep.get("missing", []))
    lines = [
        f"File: {path}",
        f"Elements: {cnt}",
        f"Namespaces: {', '.join(f'{k}={v}' for k,v in ns.items())}" if ns else "Namespaces: (none)",
    ]
    if ok_n or miss_n:
        lines.append(f"Cross-resource hrefs: ok={ok_n}, missing={miss_n}")
        if miss_n:
            # show a few missing for debugging
            miss_list = href_rep.get("missing", [])[:5]
            lines.append("Missing examples: " + "; ".join(miss_list))
    return "\n".join(lines)

# --- Export: allow saving with .model extension unchanged elsewhere ---
def _export_to_file(payload: str, file_extension: str, directory: str = ".", prefix: str = "artifact") -> str:
    """
    Export helper extended to support '.model' seamlessly.
    file_extension can be like 'xmi', '.xmi', 'model', '.model', etc.
    """
    os.makedirs(directory, exist_ok=True)
    ext = file_extension if file_extension.startswith(".") else f".{file_extension}"
    # normalize to lower-case but keep '.model' as is
    ext = ext.lower()
    fname = f"{prefix}{ext}"
    fpath = os.path.join(directory, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(payload)
    return os.path.abspath(fpath)

# --- Tiny XML helpers ---
def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def _ns_uri(tag: str) -> Optional[str]:
    if tag.startswith("{"):
        return tag.split("}", 1)[0][1:]
    return None

def _collect_nsmap(root: ET.Element) -> Dict[str, str]:
    nsmap: Dict[str, str] = {}
    # ElementTree stores xmlns:* as plain attributes on the root
    for k, v in root.attrib.items():
        if k.startswith("xmlns:"):
            nsmap[k.split(":", 1)[1]] = v
        elif k == "xmlns":
            nsmap[""] = v
    return nsmap
