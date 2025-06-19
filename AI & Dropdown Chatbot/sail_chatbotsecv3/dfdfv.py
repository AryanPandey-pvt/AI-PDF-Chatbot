import pdfplumber
import re
import json
from difflib import SequenceMatcher


def parse_pdf_to_json(pdf_path, output_json_path):
    # Extract all text from the PDF
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Improved regex pattern for multi-level headings
    heading_pattern = re.compile(r'^(\d+(?:\.\d+)+)\s+([A-Z].+)', re.MULTILINE)

    # Split by headings/subheadings, keeping the headings
    splits = heading_pattern.split(text)

    # Build the hierarchical JSON
    result = {}
    current_main = None

    i = 1
    while i < len(splits):
        num = splits[i].strip()
        title = splits[i + 1].strip()
        content = splits[i + 2].strip() if (i + 2) < len(splits) else ""

        # Determine heading level
        if re.match(r'^\d+\.0$', num):  # Main heading
            current_main = num
            result[current_main] = {
                "title": title,
                "content": content,
                "subheadings": {}
            }
        elif current_main:  # Only process subheadings if main exists
            if re.match(r'^\d+\.\d+$', num):  # Subheading
                result[current_main]["subheadings"][num] = {
                    "title": title,
                    "content": content,
                    "subsubheadings": {}
                }
            elif re.match(r'^\d+\.\d+\.\d+', num):  # Sub-subheading
                # Get parent subheading
                parent_num = ".".join(num.split(".")[:2])
                if parent_num in result[current_main]["subheadings"]:
                    result[current_main]["subheadings"][parent_num]["subsubheadings"][num] = {
                        "title": title,
                        "content": content
                    }
                else:
                    print(f"Warning: Orphan sub-subheading {num} - no parent {parent_num} found")
        i += 3

    # Save to output JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def search_pdf(pdf_path, query, fuzzy_threshold=0.7, include_content=True):
    # Parse PDF and get structure (using temporary output file)
    document = parse_pdf_to_json(pdf_path, "temp_search_output.json")
    query = query.strip().lower()
    results = []
    exact_found = False

    # Helper function to check matches
    def add_result(section, heading_num, match_type, confidence):
        nonlocal exact_found
        results.append({
            "heading": heading_num,
            "title": section["title"],
            "content": section["content"][:500] + "..." if include_content else "",
            "match_type": match_type,
            "confidence": confidence
        })
        if match_type == "exact":
            exact_found = True

    # Search through all document levels
    for main_num, main_sec in document.items():
        # Check main heading
        if main_num.lower() == query:
            add_result(main_sec, main_num, "exact", 1.0)
        if query in main_sec["title"].lower() or \
                (include_content and query in main_sec["content"].lower()):
            add_result(main_sec, main_num, "exact", 1.0)

        # Check subheadings
        for sub_num, sub_sec in main_sec["subheadings"].items():
            if sub_num.lower() == query:
                add_result(sub_sec, sub_num, "exact", 1.0)
            if query in sub_sec["title"].lower() or \
                    (include_content and query in sub_sec["content"].lower()):
                add_result(sub_sec, sub_num, "exact", 1.0)

            # Check subsubheadings
            for ssub_num, ssub_sec in sub_sec["subsubheadings"].items():
                if ssub_num.lower() == query:
                    add_result(ssub_sec, ssub_num, "exact", 1.0)
                if query in ssub_sec["title"].lower() or \
                        (include_content and query in ssub_sec["content"].lower()):
                    add_result(ssub_sec, ssub_num, "exact", 1.0)

    # Fuzzy search if no exact matches
    if not exact_found:
        all_sections = []
        for main_num, main_sec in document.items():
            all_sections.append((main_num, main_sec))
            for sub_num, sub_sec in main_sec["subheadings"].items():
                all_sections.append((sub_num, sub_sec))
                for ssub_num, ssub_sec in sub_sec["subsubheadings"].items():
                    all_sections.append((ssub_num, ssub_sec))

        for heading_num, section in all_sections:
            search_text = section["title"].lower()
            if include_content:
                search_text += " " + section["content"].lower()

            similarity = SequenceMatcher(None, query, search_text).ratio()
            if similarity >= fuzzy_threshold:
                add_result(section, heading_num, "fuzzy", similarity)

    # Sort results by confidence and heading depth
    return sorted(
        results,
        key=lambda x: (-x["confidence"], x["heading"].count(".")),
        reverse=False
    )


# Usage example
if __name__ == "__main__":
    # Parse and save structure
    parse_pdf_to_json("pcp.pdf", "pcp_output.json")

    # Search example
    results = search_pdf("pcp.pdf", "part quotations")
    print(json.dumps(results[:2], indent=2))  # Print top 2 results
