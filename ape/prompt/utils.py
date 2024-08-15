import json
from ape.types import Dataset, DatasetItem
from ape.types.response_format import ResponseFormat, ResponseFormatType
from ape.utils import dict_to_xml


def format_fewshot(fewshot: Dataset, response_format: ResponseFormat) -> str:
    """Format fewshot data into specific format."""
    formatted = ""
    try:
        for data in fewshot:
            if isinstance(data, DatasetItem):
                inputs, outputs = data.inputs, data.outputs
            else:
                inputs, outputs = data["inputs"], data.get("outputs", {})
            if response_format.type == ResponseFormatType.XML:
                inputs_xml = dict_to_xml(inputs, "input")
                outputs_xml = dict_to_xml(outputs, "output")
                formatted += f"<example>\n{inputs_xml}\n{outputs_xml}\n</example>"
            else:
                formatted += f"Inputs:\n{json.dumps(inputs, indent=2)}"
                formatted += f"Outputs:\n{json.dumps(outputs, indent=2)}"
    except Exception as err:
        print(err)

    return formatted
