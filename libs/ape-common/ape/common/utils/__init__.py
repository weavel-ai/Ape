import re
import asyncio
from typing import Any, Dict, List, Optional, Union
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import warnings

from .logging import logger


def parse_xml_outputs(text: str) -> Dict[str, str]:
    warnings.warn(
        "The parse_xml_outputs function is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Find the outermost <outputs> tag pair
    match = re.search(r"\s*<outputs>(.*?)</outputs>\s*", text, re.DOTALL)
    if not match:
        logger.warning("No <outputs> tags found, attempting to parse <output> tags directly")
        output_pattern = r'<output\s+name="([^"]+)">(.*?)</output>'
        outputs = dict(re.findall(output_pattern, text, re.DOTALL))
        if not outputs:
            raise ValueError(f"No valid <output> tags found either. {repr(text)}")
        return {k: v.strip() for k, v in outputs.items()}

    content = match.group(1)

    # Function to replace nested <outputs> tags with escaped versions
    def escape_nested_outputs(match):
        return match.group(0).replace("<", "&lt;").replace(">", "&gt;")

    # Escape any nested <outputs> tags
    content = re.sub(r"<outputs>.*?</outputs>", escape_nested_outputs, content, flags=re.DOTALL)

    # Find all <output> tags within the <outputs> tag
    output_pattern = r'<output\s+name="([^"]+)">(.*?)</output>'
    outputs = dict(re.findall(output_pattern, content, re.DOTALL))

    # Unescape the content and strip whitespace
    outputs = {k: v.replace("&lt;", "<").replace("&gt;", ">").strip() for k, v in outputs.items()}

    return outputs


def dict_to_xml(data: Dict[str, Any], tag_name: str) -> str:
    warnings.warn(
        "The dict_to_xml function is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Create the root element
    root = ET.Element(f"{tag_name}s")

    # Add each output as a child element
    for name, value in data.items():
        # if key_ignore and name in key_ignore:
        #     continue
        tag = ET.SubElement(root, tag_name)
        tag.set("name", name)
        tag.text = str(value)  # Convert value to string

    # Convert the XML tree to a string
    rough_string = ET.tostring(root, encoding="unicode")

    # Use minidom to prettify the XML
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # Remove the XML declaration
    pretty_xml = "\n".join(pretty_xml.split("\n")[1:])

    # Remove extra newlines (minidom adds too many)
    pretty_xml = "\n".join(line for line in pretty_xml.split("\n") if line.strip())

    return pretty_xml
