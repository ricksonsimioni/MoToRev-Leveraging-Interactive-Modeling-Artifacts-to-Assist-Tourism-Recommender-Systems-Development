def remove_ns1_id_plain_text(input_file, output_file):
    # Read the input file
    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Remove all occurrences of `ns1:id="..."` using a regex
    import re
    cleaned_content = re.sub(r' ns1:id="[^"]*"', '', content)

    # Write the cleaned content to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(cleaned_content)

# Define the file paths
input_file = "/Users/ricksonsimionipereira/eclipse-workspace/Conferences/RecommenderSystem/genericRecommenderSystem/src/main/Models/old.xml"  # Replace with your input file name
output_file = "/Users/ricksonsimionipereira/eclipse-workspace/Conferences/RecommenderSystem/genericRecommenderSystem/src/main/Models/new.xml"  # Replace with your desired output file name

# Call the function
remove_ns1_id_plain_text(input_file, output_file)

print(f"'ns1:id' attributes removed. Updated file saved as '{output_file}'.")
