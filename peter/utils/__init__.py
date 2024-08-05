import re
import asyncio
from typing import Dict
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


def parse_xml_outputs(text: str) -> Dict[str, str]:
    match = re.search(r"<outputs>.*?</outputs>", text, re.DOTALL)
    if match:
        xml_string = match.group(0)
    else:
        raise ValueError("No valid outputs found.")
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Initialize a dictionary to hold the outputs
    outputs = {}

    # Iterate over each output element in the XML
    for out in root.findall("output"):
        name = out.get("name")  # Get the output name attribute
        value = out.text  # Get the output value
        if name:
            outputs[name] = value

    return outputs


def dict_to_xml(data: Dict[str, str], tag_name: str) -> str:
    # Create the root element
    root = ET.Element(f"{tag_name}s")

    # Add each output as a child element
    for name, value in data.items():
        tag = ET.SubElement(root, tag_name)
        tag.set("name", name)
        tag.text = value

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


def parse_numbered_list(text: str) -> list[str]:
    lines = text.strip().split("\n")
    pattern = r"^\d+\.\s*(.+)$"
    extracted_list = [
        re.match(pattern, line).group(1) for line in lines if re.match(pattern, line)
    ]
    return extracted_list


def is_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
    except ImportError:
        return False  # IPython not installed


def run_async(coroutine):
    if is_notebook():
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            print("Please install nest_asyncio: !pip install nest_asyncio")
            raise

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coroutine)
    else:
        return asyncio.run(coroutine)
