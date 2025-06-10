import random
import xml.etree.ElementTree as ET
import string

# Function to generate random XMI id
def generate_xmi_id():
    # Randomly generate the first 11 characters (uppercase letters and digits)
    random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=11))
    return f"_{random_part}Ee-3SJufpezX0Q"  # Keep the last part the same

# Function to generate rows with random ratings
def generate_rows(xmi_ids, users, items, min_rating=3.5, max_rating=5.0, ratings_per_user=8):
    rows = []

    for user in users:
        # Randomly select items for this user, allowing repetition but limiting to 5 ratings
        rated_items = random.sample(items * ratings_per_user, ratings_per_user)  # Allow repetition of items
        random.shuffle(rated_items)  # Shuffle to add randomness in order

        for item in rated_items:
            # Randomly generate a rating between min_rating and max_rating
            rating = round(random.uniform(min_rating, max_rating), 1)  # Random float with 1 decimal place

            # Generate the row XML element
            row = ET.Element("rows", {
                "xmi:id": generate_xmi_id(),  # Generate a random XMI id
                "value": str(rating)  # Assign the rating value
            })
            
            # Create the user and item elements
            user_element = ET.SubElement(row, "_user", {"href": f"domain.model#{user}"})
            item_element = ET.SubElement(row, "_item", item)  # Unpack the item dictionary

            rows.append(row)
    
    return rows

# Inputs
xmi_ids = []  # No need to define XMI ids manually anymore
users = [
    "_jzWPkBRhEe-xqaUUKrsSng", "_4iuNMBRXEe-xqaUUKrsSng", "_k-AE0BRhEe-xqaUUKrsSng", 
    "_mviT4BRhEe-xqaUUKrsSng", "_9zJlMCMiEe-kdKBnMBYA1g", "_-QDjoCMiEe-kdKBnMBYA1g", 
    "_-tOn0CMiEe-kdKBnMBYA1g", "__DaTMCMiEe-kdKBnMBYA1g", "__W2MkCMiEe-kdKBnMBYA1g", 
    "_AXysMCMjEe-kdKBnMBYA1g", "_Ap-oYCMjEe-kdKBnMBYA1g", "_ADZwkCMjEe-kdKBnMBYA1g", 
    "_A-HFUCMjEe-kdKBnMBYA1g", "_IYpCwKwTEe-3SJufpezX0Q", "_Hue2oBRhEe-xqaUUKrsSng", 
    "_hihccBRhEe-xqaUUKrsSng", "_jTkjUBRhEe-xqaUUKrsSng", "_kLTckBRhEe-xqaUUKrsSng", 
    "_kkO58BRhEe-xqaUUKrsSng", "_mY2SMBRhEe-xqaUUKrsSng"
]

items = [
    {"xsi:type": "domain:Outdoor", "href": "domain.model#_0gfqoJ3PEe-msKNMZk5ySw"},
    {"xsi:type": "domain:Outdoor", "href": "domain.model#_HZUqcJ3TEe-msKNMZk5ySw"},
    {"xsi:type": "domain:Outdoor", "href": "domain.model#_BH1o0J3UEe-msKNMZk5ySw"},
    {"xsi:type": "domain:Outdoor", "href": "domain.model#_TY2XkBRXEe-xqaUUKrsSng"},
    {"xsi:type": "domain:Outdoor", "href": "domain.model#_Ve7TQBRXEe-xqaUUKrsSng"},
    {"xsi:type": "domain:Outdoor", "href": "domain.model#_WpJDoBRXEe-xqaUUKrsSng"},
    {"xsi:type": "domain:Outdoor", "href": "domain.model#_JM6SgJ3UEe-msKNMZk5ySw"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_VI2VkBRXEe-xqaUUKrsSng"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_U5Bw4BRXEe-xqaUUKrsSng"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_C4cq0J3UEe-msKNMZk5ySw"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_FeijUJ3UEe-msKNMZk5ySw"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_CtVnQCMjEe-kdKBnMBYA1g"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_D9pYkCMjEe-kdKBnMBYA1g"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_EUDGYCMjEe-kdKBnMBYA1g"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_E3H-QCMjEe-kdKBnMBYA1g"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_uPauIJ3PEe-msKNMZk5ySw"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_mjSv8J3PEe-msKNMZk5ySw"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_qRI6sJ3PEe-msKNMZk5ySw"},
    {"xsi:type": "domain:Outdoor", "href": "domain.model#_tMPhgLC1Ee-9_-U5v1vn1w"},
    {"xsi:type": "domain:Indoor", "href": "domain.model#_X2jWULC1Ee-9_-U5v1vn1w"},
]

# Generate rows
generated_rows = generate_rows(xmi_ids, users, items)

# Create the XML tree
root = ET.Element("data", xmlns="http://www.w3.org/2001/XMLSchema-instance")  # Add namespace declaration
for row in generated_rows:
    root.append(row)

# Write to file with indentation
xml_str = ET.tostring(root, encoding="unicode", method="xml")
xml_str = '<?xml version="1.0" encoding="ASCII"?>\n' + xml_str

# Pretty indentation for readability
#from xml.dom import minidom
#xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")

with open("parcoNazionale.xml", "w") as file:
    file.write(xml_str)

print("XML file generated with randomized rows and beautiful indentation.")
