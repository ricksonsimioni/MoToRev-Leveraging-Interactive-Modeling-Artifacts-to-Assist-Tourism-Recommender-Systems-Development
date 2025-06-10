import uuid
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def generate_id()  :
    random_part = uuid.uuid4().int[:11]  # Generate a random 11-character string
    return f"_{random_part}"

def generate_xml_with_fixed_suffix(dataframe):
    """Converts a DataFrame into the specified XML structure with fixed XMI ID suffix."""
    root = Element("domainMovie:MovieDomain", {
        "xmi:version": "2.0",
        "xmlns:xmi": "http://www.omg.org/XMI",
        "xmlns:domainMovie": "http://org.rs.domain.movie",
        "xmi:id": generate_xmi_id_with_suffix(),
        "name": "Flix"
    })

    # Generate categories and map them to IDs
    unique_genres = set("|".join(dataframe["genres"]).split("|"))
    genre_to_id = {genre: generate_xmi_id_with_suffix() for genre in unique_genres if genre}
    
    # Add categories to the XML
    for genre, genre_id in genre_to_id.items():
        SubElement(root, "categories", {"xmi:id": genre_id, "category": genre})

    # Add movies to the XML
    for index, row in dataframe.iterrows():
        movie_element = SubElement(root, "movies", {
            "xmi:id": generate_xmi_id_with_suffix(),
            "identification": str(index + 1),
            "category": genre_to_id[row["genres"].split("|")[0]],  # Use the first genre as the category
            "name": row["title"]
        })

    # Pretty print the XML
    return parseString(tostring(root)).toprettyxml(indent="  ")

# Generate the XML using the new ID logic
xml_output_with_suffix = generate_xml_with_fixed_suffix(movies_data)

# Save the new XML to a file
output_file_with_suffix = '/mnt/data/movies_converted_with_suffix.xml'
with open(output_file_with_suffix, 'w', encoding='utf-8') as file:
    file.write(xml_output_with_suffix)

output_file_with_suffix
