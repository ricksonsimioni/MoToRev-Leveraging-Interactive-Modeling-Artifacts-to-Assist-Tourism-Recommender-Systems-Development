import xml.etree.ElementTree as ET
import random
import string

# Paths for files (use correct paths as per the session context)
corrected_file_path = '/Users/ricksonsimionipereira/eclipse-workspace/Conferences/RecommenderSystem/genericRecommenderSystem/domain.movie.model'
#ratings_output_path = '/mnt/data/ratings_generated.xml'

# Reload the corrected domain.movie.model XML
tree_corrected = ET.parse(corrected_file_path)
root_corrected = tree_corrected.getroot()

# Reload IDs
viewers_corrected = {viewer.get('name'): viewer.get('{http://www.omg.org/XMI}id') for viewer in root_corrected.findall('.//Viewer')}
movies_corrected = {movie.get('name'): movie.get('{http://www.omg.org/XMI}id') for movie in root_corrected.findall('.//movies')}
series_corrected = {serie.get('name'): serie.get('{http://www.omg.org/XMI}id') for serie in root_corrected.findall('.//series')}
categories_corrected = {category.get('category'): category.get('{http://www.omg.org/XMI}id') for category in root_corrected.findall('.//categories') if category.get('category')}

# Generate random rows
def random_xmi_id():
    """Generate a randomized XMI ID keeping the '-kdKBnMBYA1g' suffix."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + "-kdKBnMBYA1g"

rows_data = []
cb_preferences = []
ratings_random = random.randint(150, 250)

for viewer_name, viewer_id in viewers_corrected.items():
    # Generate 15 random ratings for each viewer
    rated_items = random.sample(list(movies_corrected.values()) + list(series_corrected.values()), ratings_random)
    for item_id in rated_items:
        item_type = "domainMovie:Movie" if item_id in movies_corrected.values() else "domainMovie:Series"
        rows_data.append({
            "row_id": random_xmi_id(),
            "user_id": viewer_id,
            "item_id": item_id,
            "item_type": item_type,
            "value": round(random.uniform(3.5, 5.0), 1)
        })

    # Assign a random category preference to each viewer
    preferred_category = random.choice(list(categories_corrected.values()))
    cb_preferences.append({
        "preference_id": random_xmi_id(),
        "user_id": viewer_id,
        "category_id": preferred_category
    })

# Create the XML structure
algorithm = ET.Element("org.rs:Algorithm", {
    "xmi:version": "2.0",
    "xmlns:xmi": "http://www.omg.org/XMI",
    "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "xmlns:domainMovie": "http://org.rs.domain.movie",
    "xmlns:org.rs": "http://org.rs",
    "xmi:id": "_SWHPICEDEe-kdKBnMBYA1g",
    "name": "MovieLens"
})

filtering_rs = ET.SubElement(algorithm, "filteringRS", {
    "xsi:type": "org.rs:HybridBased",
    "xmi:id": "_aV7lYCEDEe-kdKBnMBYA1g"
})

cf_component = ET.SubElement(filtering_rs, "_cfComponent", {"xmi:id": "_bTnVoCEDEe-kdKBnMBYA1g"})
data = ET.SubElement(cf_component, "data", {"xmi:id": "_T8G8ICF9Ee-kdKBnMBYA1g"})

for row in rows_data:
    row_element = ET.SubElement(data, "rows", {
        "xmi:id": row["row_id"],
        "value": str(row["value"])
    })
    user_element = ET.SubElement(row_element, "_user", {"href": f"domain.movie.model#{row['user_id']}"})
    item_element = ET.SubElement(row_element, "_item", {
        "xsi:type": row["item_type"],
        "href": f"domain.movie.model#{row['item_id']}"
    })

cb_component = ET.SubElement(filtering_rs, "_cbComponent", {"xmi:id": "_anugACEDEe-kdKBnMBYA1g"})
for pref in cb_preferences:
    pref_element = ET.SubElement(cb_component, "contentBasedPreference", {"xmi:id": pref["preference_id"]})
    user_element = ET.SubElement(pref_element, "_user", {"href": f"domain.movie.model#{pref['user_id']}"})
    prefs_element = ET.SubElement(pref_element, "_prefs", {
        "xsi:type": "domainMovie:ShowCategory",
        "href": f"domain.movie.model#{pref['category_id']}"
    })

# Write the generated XML to a file
ratings_output_path = "/Users/ricksonsimionipereira/eclipse-workspace/Conferences/RecommenderSystem/genericRecommenderSystem/ratings_generated-movie.xml"
ET.ElementTree(algorithm).write(ratings_output_path, encoding="ASCII", xml_declaration=True)

print(f"XML file has been generated and saved to: {ratings_output_path}")
