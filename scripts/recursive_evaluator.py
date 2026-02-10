import xml.etree.ElementTree as ET
import re
import os
import glob
from typing import Dict, List, Set, Tuple
from datetime import datetime
from collections import defaultdict


class ModelEvaluator:
    """Evaluates models against metamodels for correctness and hallucination rate"""

    def __init__(self, domain_metamodel_path: str, rs_metamodel_path: str):
        """Initialize with paths to metamodel files"""
        self.domain_metamodel = self._load_xml(domain_metamodel_path)
        self.rs_metamodel = self._load_xml(rs_metamodel_path)

        # Extract valid types from metamodels
        self.valid_domain_classes = self._extract_class_names(self.domain_metamodel)
        self.valid_rs_classes = self._extract_class_names(self.rs_metamodel)

        # Extract valid enum values
        self.valid_domain_enums = self._extract_enum_values(self.domain_metamodel)
        self.valid_rs_enums = self._extract_enum_values(self.rs_metamodel)

        print(f"Loaded {len(self.valid_domain_classes)} domain classes and {len(self.valid_domain_enums)} enums")

    def _load_xml(self, path: str) -> ET.Element:
        """Load XML file"""
        tree = ET.parse(path)
        return tree.getroot()

    def _extract_class_names(self, metamodel: ET.Element) -> Set[str]:
        """Extract all EClass names from metamodel"""
        classes = set()
        for elem in metamodel.iter():
            if 'EClass' in elem.tag:
                name = elem.get('name')
                if name:
                    classes.add(name)
        return classes

    def _extract_enum_values(self, metamodel: ET.Element) -> Dict[str, List[str]]:
        """Extract all enum types and their literal values"""
        enums = {}
        for elem in metamodel.iter():
            if 'EEnum' in elem.tag:
                enum_name = elem.get('name')
                if enum_name:
                    literals = []
                    for literal in elem:
                        if 'eLiterals' in literal.tag:
                            lit_name = literal.get('name')
                            if lit_name:
                                literals.append(lit_name)
                    if literals:
                        enums[enum_name] = literals
        return enums

    def evaluate_domain_model(self, domain_model_path: str) -> Dict:
        """Evaluate a domain model instance"""
        model = self._load_xml(domain_model_path)
        return self._analyze_domain_model(model)

    def _analyze_domain_model(self, model: ET.Element) -> Dict:
        """Analyze domain model structure with detailed error reporting"""
        results = {
            'tourists': {},
            'pois': {},
            'categories': {},
            'issues': [],
            'issue_details': {
                'type_errors': [],
                'missing_required': [],
                'invalid_references': [],
                'invalid_enum_values': []
            },
            'is_correct': True
        }

        # Define namespace
        ns = '{http://org.rs.domain}'

        # Find all tourists - tag is {http://org.rs.domain}Tourist
        for elem in model.iter():
            if elem.tag == f'{ns}Tourist':
                elem_id = elem.get('{http://www.omg.org/XMI}id')

                if elem_id:
                    results['tourists'][elem_id] = {
                        'type': 'Tourist',
                        'name': elem.get('name'),
                        'element': elem.tag
                    }

        # Find all POIs - tag is {http://org.rs.domain}IndoorPOI or OutdoorPOI
        for elem in model.iter():
            if elem.tag in [f'{ns}IndoorPOI', f'{ns}OutdoorPOI']:
                elem_id = elem.get('{http://www.omg.org/XMI}id')

                if elem_id:
                    poi_type = elem.tag.replace(ns, '')  # Extract 'IndoorPOI' or 'OutdoorPOI'
                    results['pois'][elem_id] = {
                        'type': poi_type,
                        'name': elem.get('name'),
                        'element': elem.tag
                    }

                    # Validate it's one of the valid types
                    valid_poi_types = ['IndoorPOI', 'OutdoorPOI']
                    if poi_type not in valid_poi_types:
                        error = {
                            'element_id': elem_id,
                            'element_name': elem.get('name'),
                            'found_type': poi_type,
                            'expected_types': valid_poi_types
                        }
                        results['issue_details']['type_errors'].append(error)
                        results['issues'].append(
                            f"POI {elem_id}: Invalid type '{poi_type}'. Expected IndoorPOI or OutdoorPOI"
                        )
                        results['is_correct'] = False

        # Find all categories - tag is {http://org.rs.domain}Category
        for elem in model.iter():
            if elem.tag == f'{ns}Category':
                elem_id = elem.get('{http://www.omg.org/XMI}id')
                category_attr = elem.get('category')

                if elem_id:
                    results['categories'][elem_id] = {
                        'type': 'Category',
                        'category': category_attr,
                        'element': elem.tag
                    }

                    # Validate category enum value
                    if category_attr and 'POICategory' in self.valid_domain_enums:
                        valid_categories = self.valid_domain_enums['POICategory']
                        if category_attr not in valid_categories:
                            error = {
                                'element_id': elem_id,
                                'found_value': category_attr,
                                'expected_values': valid_categories
                            }
                            results['issue_details']['invalid_enum_values'].append(error)
                            results['issues'].append(
                                f"Category {elem_id}: Invalid value '{category_attr}'. "
                                f"Valid: {', '.join(valid_categories)}"
                            )
                            results['is_correct'] = False

        return results

    def evaluate_rs_model(self, rs_model_path: str, domain_model_path: str) -> Dict:
        """Evaluate RS model against domain model"""
        rs_model = self._load_xml(rs_model_path)
        domain_model = self._load_xml(domain_model_path)
        domain_analysis = self._analyze_domain_model(domain_model)

        return self._analyze_rs_model(rs_model, domain_analysis)

    def _analyze_rs_model(self, rs_model: ET.Element, domain_analysis: Dict) -> Dict:
        """Analyze RS model for hallucinations with detailed reporting"""
        results = {
            'algorithm_type': None,
            'rows': [],
            'referenced_users': set(),
            'referenced_items': set(),
            'hallucinated_users': set(),
            'hallucinated_items': set(),
            'hallucination_details': {
                'user_hallucinations': [],
                'item_hallucinations': [],
                'reference_format_issues': []
            },
            'issues': [],
            'is_correct': True
        }

        # Detect algorithm type from root element
        rs_ns = '{http://org.rs}'
        for elem in rs_model.iter():
            if elem.tag in [f'{rs_ns}CollaborativeFiltering', f'{rs_ns}CollaborativeBased', 
                           f'{rs_ns}ContentBased', f'{rs_ns}HybridBased']:
                results['algorithm_type'] = elem.tag.replace(rs_ns, '')
                break

        # Extract references based on algorithm type
        # For CollaborativeFiltering: Look for UserItemRow in rows
        for elem in rs_model.iter():
            if elem.tag == 'rows' or 'rows' in elem.tag.lower():
                # This is a UserItemRow element
                user_ref = elem.get('_user') or elem.get('user')
                item_ref = elem.get('_item') or elem.get('item')

                if user_ref:
                    results['referenced_users'].add(user_ref)
                if item_ref:
                    results['referenced_items'].add(item_ref)

                results['rows'].append({
                    'user': user_ref,
                    'item': item_ref,
                    'value': elem.get('value'),
                    'row_id': elem.get('{http://www.omg.org/XMI}id')
                })

        # For ContentBased: Look for ContentBasedPreference elements
        for elem in rs_model.iter():
            if 'ContentBasedPreference' in elem.tag or elem.tag == 'contentBasedPreference':
                user_ref = elem.get('_user') or elem.get('user')

                if user_ref:
                    results['referenced_users'].add(user_ref)

                # Look for _prefs references (to preferences/categories)
                prefs_ref = elem.get('_prefs') or elem.get('prefs')
                if prefs_ref:
                    # ContentBased references preferences, not items directly
                    # We'll skip item hallucination check for ContentBased
                    pass

        # For HybridBased: Process both components
        # (Will be handled by recursive processing of child elements)

        # Check for hallucinations
        real_tourists = set(domain_analysis['tourists'].keys())
        real_pois = set(domain_analysis['pois'].keys())

        # Extract IDs from references
        def extract_id(ref: str) -> str:
            if not ref:
                return ""
            # Handle formats like:
            # - "domain.model#_gen_tourist_1"
            # - "../domain.model#_gen_poi_5"
            # - "_gen_tourist_1" (direct reference)
            if '#' in ref:
                return ref.split('#')[-1]
            if '/' in ref:
                return ref.split('/')[-1]
            return ref

        referenced_user_ids = {extract_id(ref) for ref in results['referenced_users'] if ref}
        referenced_item_ids = {extract_id(ref) for ref in results['referenced_items'] if ref}

        # Remove empty strings
        referenced_user_ids.discard("")
        referenced_item_ids.discard("")

        # Find hallucinations
        results['hallucinated_users'] = referenced_user_ids - real_tourists
        results['hallucinated_items'] = referenced_item_ids - real_pois

        # Detailed user hallucination analysis
        if results['hallucinated_users']:
            for user_id in sorted(results['hallucinated_users']):
                referencing_rows = [r for r in results['rows'] if extract_id(r.get('user', '')) == user_id]

                hallucination = {
                    'hallucinated_id': user_id,
                    'reference_count': len(referencing_rows),
                    'sample_references': [r['user'] for r in referencing_rows[:3] if r.get('user')]
                }
                results['hallucination_details']['user_hallucinations'].append(hallucination)

            results['issues'].append(
                f"User Hallucinations: RS model references {len(results['hallucinated_users'])} "
                f"users not in domain. Sample: {', '.join(sorted(list(results['hallucinated_users']))[:5])}"
            )
            results['is_correct'] = False

        # Detailed item hallucination analysis
        if results['hallucinated_items']:
            for item_id in sorted(results['hallucinated_items']):
                referencing_rows = [r for r in results['rows'] if extract_id(r.get('item', '')) == item_id]

                hallucination = {
                    'hallucinated_id': item_id,
                    'reference_count': len(referencing_rows),
                    'sample_references': [r['item'] for r in referencing_rows[:3] if r.get('item')]
                }
                results['hallucination_details']['item_hallucinations'].append(hallucination)

            # Analyze pattern
            real_poi_nums = set()
            halluc_poi_nums = set()
            for poi_id in real_pois:
                match = re.search(r'_poi_(\d+)', poi_id)
                if match:
                    real_poi_nums.add(int(match.group(1)))
            for poi_id in results['hallucinated_items']:
                match = re.search(r'_poi_(\d+)', poi_id)
                if match:
                    halluc_poi_nums.add(int(match.group(1)))

            if real_poi_nums and halluc_poi_nums:
                max_real = max(real_poi_nums)
                min_halluc = min(halluc_poi_nums) if halluc_poi_nums else 0
                if min_halluc > max_real:
                    results['issues'].append(
                        f"Item Hallucinations: RS references POIs {min_halluc}-{max(halluc_poi_nums)} "
                        f"but domain only has 1-{max_real}. Total: {len(results['hallucinated_items'])}"
                    )
                else:
                    results['issues'].append(
                        f"Item Hallucinations: {len(results['hallucinated_items'])} items not in domain. "
                        f"Sample: {', '.join(sorted(list(results['hallucinated_items']))[:5])}"
                    )
            else:
                results['issues'].append(
                    f"Item Hallucinations: {len(results['hallucinated_items'])} items not in domain. "
                    f"Sample: {', '.join(sorted(list(results['hallucinated_items']))[:5])}"
                )
            results['is_correct'] = False

        return results

    def compute_correctness(self, domain_results: List[Dict], rs_results: List[Dict]) -> float:
        """Compute Correctness metric"""
        total_models = len(domain_results) + len(rs_results)
        correct_models = sum(1 for r in domain_results if r['is_correct'])
        correct_models += sum(1 for r in rs_results if r['is_correct'])

        return correct_models / total_models if total_models > 0 else 0.0

    def compute_hallucination_rate(self, rs_results: Dict, domain_results: Dict) -> Dict:
        """Compute Hallucination Rate"""
        real_users = len(domain_results['tourists'])
        real_items = len(domain_results['pois'])

        generated_users = len(rs_results['referenced_users'])
        generated_items = len(rs_results['referenced_items'])

        user_rate = generated_users / real_users if real_users > 0 else float('inf')
        item_rate = generated_items / real_items if real_items > 0 else float('inf')

        return {
            'user_hallucination_rate': user_rate,
            'item_hallucination_rate': item_rate,
            'real_users': real_users,
            'real_items': real_items,
            'generated_users': generated_users,
            'generated_items': generated_items,
            'hallucinated_users': len(rs_results['hallucinated_users']),
            'hallucinated_items': len(rs_results['hallucinated_items'])
        }

    def evaluate_directory(self, directory_path: str) -> Dict:
        """Evaluate all model pairs in a directory"""
        patterns = [
            ("domain-*.model", "recommender-*.model"),
            ("*domain*.model", "*recommender*.model"),
            ("*.domain.model", "*.recommender.model"),
        ]

        domain_files = []
        rs_files = []

        for domain_pattern, rs_pattern in patterns:
            domain_files = sorted(glob.glob(os.path.join(directory_path, domain_pattern)))
            rs_files = sorted(glob.glob(os.path.join(directory_path, rs_pattern)))
            if domain_files and rs_files:
                break

        results = {
            'directory': directory_path,
            'model_pairs': [],
            'domain_results': [],
            'rs_results': [],
            'overall_correctness': 0.0,
            'avg_user_hallucination': 0.0,
            'avg_item_hallucination': 0.0,
            'total_pairs': 0,
            'errors': []
        }

        num_pairs = min(len(domain_files), len(rs_files))
        results['total_pairs'] = num_pairs

        if num_pairs == 0:
            return results

        for i in range(num_pairs):
            domain_file = domain_files[i]
            rs_file = rs_files[i]

            try:
                domain_result = self.evaluate_domain_model(domain_file)
                rs_result = self.evaluate_rs_model(rs_file, domain_file)
                hallucination = self.compute_hallucination_rate(rs_result, domain_result)

                results['domain_results'].append(domain_result)
                results['rs_results'].append(rs_result)

                results['model_pairs'].append({
                    'domain_file': os.path.basename(domain_file),
                    'rs_file': os.path.basename(rs_file),
                    'domain_correct': domain_result['is_correct'],
                    'rs_correct': rs_result['is_correct'],
                    'domain_issues': domain_result['issues'],
                    'rs_issues': rs_result['issues'],
                    'domain_issue_details': domain_result['issue_details'],
                    'rs_hallucination_details': rs_result['hallucination_details'],
                    'hallucination': hallucination
                })

            except Exception as e:
                import traceback
                results['errors'].append({
                    'domain_file': os.path.basename(domain_file),
                    'rs_file': os.path.basename(rs_file),
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })

        # Compute overall metrics
        if results['domain_results'] and results['rs_results']:
            results['overall_correctness'] = self.compute_correctness(
                results['domain_results'], 
                results['rs_results']
            )

            user_rates = [p['hallucination']['user_hallucination_rate'] 
                         for p in results['model_pairs'] 
                         if p['hallucination']['user_hallucination_rate'] != float('inf')]
            item_rates = [p['hallucination']['item_hallucination_rate'] 
                         for p in results['model_pairs'] 
                         if p['hallucination']['item_hallucination_rate'] != float('inf')]

            results['avg_user_hallucination'] = sum(user_rates) / len(user_rates) if user_rates else 0
            results['avg_item_hallucination'] = sum(item_rates) / len(item_rates) if item_rates else 0

        return results


def find_model_directories(base_path: str) -> List[Tuple[str, str]]:
    """Recursively find all directories containing model files"""
    model_dirs = []

    for root, dirs, files in os.walk(base_path):
        has_domain = any('domain' in f.lower() and f.endswith('.model') for f in files)
        has_recommender = any('recommender' in f.lower() and f.endswith('.model') for f in files)

        if has_domain and has_recommender:
            relative_path = os.path.relpath(root, base_path)
            scenario_name = relative_path.replace(os.sep, '_')
            model_dirs.append((root, scenario_name))

    return model_dirs


def generate_report(scenario_name: str, results: Dict, output_path: str):
    """Generate detailed report"""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"MODEL EVALUATION REPORT: {scenario_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directory: {results['directory']}\n\n")

        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model pairs: {results['total_pairs']}\n")
        f.write(f"Successfully evaluated: {len(results['model_pairs'])}\n")
        f.write(f"Errors: {len(results['errors'])}\n\n")

        f.write("METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Correctness: {results['overall_correctness']:.2%}\n")
        f.write(f"Avg User Hallucination Rate: {results['avg_user_hallucination']:.4f}\n")
        f.write(f"Avg Item Hallucination Rate: {results['avg_item_hallucination']:.4f}\n\n")

        if results['model_pairs']:
            f.write("=" * 80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for idx, pair in enumerate(results['model_pairs'], 1):
                f.write(f"Pair {idx}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Domain: {pair['domain_file']}\n")
                f.write(f"RS: {pair['rs_file']}\n\n")

                f.write(f"Domain: {'✓ CORRECT' if pair['domain_correct'] else '✗ INCORRECT'}\n")
                if pair['domain_issues']:
                    for issue in pair['domain_issues']:
                        f.write(f"  - {issue}\n")
                f.write("\n")

                f.write(f"RS: {'✓ CORRECT' if pair['rs_correct'] else '✗ INCORRECT'}\n")
                if pair['rs_issues']:
                    for issue in pair['rs_issues']:
                        f.write(f"  - {issue}\n")
                f.write("\n")

                hall = pair['hallucination']
                f.write("Hallucination Metrics:\n")
                f.write(f"  Users: {hall['user_hallucination_rate']:.4f} ")
                f.write(f"({hall['generated_users']}/{hall['real_users']}, ")
                f.write(f"{hall['hallucinated_users']} hallucinated)\n")
                f.write(f"  Items: {hall['item_hallucination_rate']:.4f} ")
                f.write(f"({hall['generated_items']}/{hall['real_items']}, ")
                f.write(f"{hall['hallucinated_items']} hallucinated)\n\n")


def batch_evaluate_all_scenarios(base_path: str, domain_metamodel: str, rs_metamodel: str, output_dir: str):
    """Batch evaluate all scenarios"""
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing evaluator...")
    evaluator = ModelEvaluator(domain_metamodel, rs_metamodel)

    print(f"Scanning: {base_path}")
    model_dirs = find_model_directories(base_path)
    print(f"Found {len(model_dirs)} directories\n")

    all_results = {}

    for dir_path, scenario_name in sorted(model_dirs):
        print(f"Evaluating: {scenario_name}")

        try:
            results = evaluator.evaluate_directory(dir_path)
            all_results[scenario_name] = results

            report_path = os.path.join(output_dir, f"evaluation_{scenario_name}.txt")
            generate_report(scenario_name, results, report_path)

            print(f"  Pairs: {results['total_pairs']}, "
                  f"Correctness: {results['overall_correctness']:.0%}, "
                  f"User Hall: {results['avg_user_hallucination']:.2f}, "
                  f"Item Hall: {results['avg_item_hallucination']:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[scenario_name] = {'error': str(e)}

    # Summary report
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Scenarios: {len(model_dirs)}\n\n")

        organized = defaultdict(dict)
        for name, result in all_results.items():
            category = 'other'
            for part in name.split('_'):
                if part in ['small', 'medium']:
                    category = part
                    break
            organized[category][name] = result

        for category in sorted(organized.keys()):
            f.write(f"\n{category.upper()}\n")
            f.write("-" * 80 + "\n")
            for name in sorted(organized[category].keys()):
                result = organized[category][name]
                if 'error' in result:
                    f.write(f"  {name}: ERROR\n")
                elif result.get('total_pairs', 0) == 0:
                    f.write(f"  {name}: No pairs\n")
                else:
                    f.write(f"  {name}:\n")
                    f.write(f"    Correctness: {result['overall_correctness']:.2%}\n")
                    f.write(f"    User Hall: {result['avg_user_hallucination']:.4f}\n")
                    f.write(f"    Item Hall: {result['avg_item_hallucination']:.4f}\n")

    print(f"\nComplete! Reports in: {output_dir}")
    return all_results


if __name__ == "__main__":
    BASE_PATH = "/Users/ricksonsimionipereira/Documents/PhD/Publications/ACM TORS 2025/Motorev_datasets/tests"
    DOMAIN_METAMODEL = "/Users/ricksonsimionipereira/eclipse-workspace/Conferences/MoToRev-Park/src/main/models/domain.ecore"
    RS_METAMODEL = "/Users/ricksonsimionipereira/eclipse-workspace/Conferences/MoToRev-Park/src/main/models/recommendersystemGeneric.ecore"
    OUTPUT_DIR = os.path.join(BASE_PATH, "evaluation_reports")

    print("=" * 80)
    print("FINAL WORKING MODEL EVALUATOR")
    print("=" * 80)
    print("Based on actual XML structure analysis")
    print()

    batch_evaluate_all_scenarios(BASE_PATH, DOMAIN_METAMODEL, RS_METAMODEL, OUTPUT_DIR)