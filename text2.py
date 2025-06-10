import itertools
import random
import string
import xml.etree.ElementTree as ET

# Function to generate random IDs with fixed suffix
def randomize_xmi_id(suffix, length=11):
    prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return f"_{prefix}{suffix}"

# Function to extract users and items from XML snippet
def extract_users_items(xml_snippet):
    root = ET.fromstring(f"<data>{xml_snippet}</data>")
    users = set()
    items = set()

    for row in root.findall("rows"):
        user_href = row.find("_user").attrib["href"].split("#")[1]
        item_href = row.find("_item").attrib["href"].split("#")[1]
        users.add(user_href)
        items.add(item_href)

    return list(users), list(items)

# Function to generate combinations and randomize the parameters
# Function to generate rows
def generate_rows(xmi_ids, users, items):
    rows = []
    for user in users:
        # Select 5 random items for each user
        rated_items = random.sample(items, 5)
        for i, item in enumerate(rated_items):
            # Generate a unique xmi_id for each row
            xmi_id = random.choice(xmi_ids)
            # Generate a random rating value between 3.5 and 5.0
            value = round(random.uniform(3.5, 5.0), 1)
            
            # Create the XML element for a row
            row = ET.Element("rows", {
                "xmi:id": xmi_id,
                "value": f"{value:.1f}"  # Formatting float to 1 decimal place
            })
            # Add user and item as sub-elements
            user_element = ET.SubElement(row, "_user", {"href": f"domain.model#{user}"})
            item_element = ET.SubElement(row, "_item", item)
            
            rows.append(row)
    return rows
# Inputs
xmi_suffix = "KwOEe-3SJufpezX0Q"
values = [1.0, 2.5, 3.9, 4.2, 5.0]
xml_snippet = """
<rows xmi:id="_H2_ggKwOEe-3SJufpezX0Q" value="3.9">
    <_user href="domain.model#_jzWPkBRhEe-xqaUUKrsSng"/>
    <_item xsi:type="domain:Outdoor" href="domain.model#_0gfqoJ3PEe-msKNMZk5ySw"/>
</rows>
<rows xmi:id="_IU3vUKwOEe-3SJufpezX0Q" value="4.2">
    <_user href="domain.model#_4iuNMBRXEe-xqaUUKrsSng"/>
    <_item xsi:type="domain:Outdoor" href="domain.model#_HZUqcJ3TEe-msKNMZk5ySw"/>
</rows>
"""

# Extract users and items
users, items = extract_users_items(xml_snippet)

# Generate rows
generated_rows = generate_rows(xmi_suffix, values, users, items)

# Build XML tree
root = ET.Element("data")
for row in generated_rows:
    root.append(row)

# Output XML to file
xml_tree = ET.ElementTree(root)
with open("paicrem.xml", "wb") as file:
    xml_tree.write(file, encoding="ASCII", xml_declaration=True)

print("XML file generated with randomized rows.")
