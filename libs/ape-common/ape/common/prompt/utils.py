import json
from typing import List, Optional
from ape.common.types import DatasetItem
from ape.common.types.response_format import ResponseFormat
from ape.common.utils import dict_to_xml


def format_fewshot(
    fewshot: List[DatasetItem],
    response_format: Optional[ResponseFormat] = None,
    # input_key_ignore: Optional[List[str]] = None,
    # output_key_ignore: Optional[List[str]] = None
) -> str:
    """Format fewshot data into specific format."""
    formatted = ""
    try:
        for idx, data in enumerate(fewshot):
            formatted += f"### Demo {idx+1} ###\n"
            inputs, outputs = data["inputs"], data.get("outputs", {})
            formatted += "**Inputs**\n"
            for key, value in inputs.items():
                # if input_key_ignore and key in input_key_ignore:
                #     continue
                formatted += f"{key.capitalize()}:\n{value}\n"
            # if output_key_ignore:
            #     for key in output_key_ignore:
            #         outputs.pop(key, None)
            formatted += f"**Outputs**\n{json.dumps(outputs, indent=2)}\n\n"
    except Exception as err:
        print(err)

    return formatted
