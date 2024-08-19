import json
from typing import Optional
from ape.types import Dataset, DatasetItem
from ape.types.response_format import ResponseFormat
from ape.utils import dict_to_xml


def format_fewshot(
    fewshot: Dataset, response_format: Optional[ResponseFormat] = None
) -> str:
    """Format fewshot data into specific format."""
    formatted = ""
    try:
        for idx, data in enumerate(fewshot):
            formatted += f"### Demo {idx+1} ###\n"
            if isinstance(data, DatasetItem):
                inputs, outputs = data.inputs, data.outputs
            else:
                inputs, outputs = data["inputs"], data.get("outputs", {})
            if response_format is not None:
                if response_format.type == "xml":
                    inputs_xml = dict_to_xml(inputs, "input")
                    outputs_xml = dict_to_xml(outputs, "output")
                    formatted += f"<example>\n{inputs_xml}\n{outputs_xml}\n</example>\n"
                    return formatted
            formatted += "**Inputs**\n"
            for key, value in inputs.items():
                formatted += f"{key.capitalize()}:\n{value}\n"
            formatted += f"**Outputs**\n{json.dumps(outputs, indent=2)}\n\n"
    except Exception as err:
        print(err)

    return formatted
