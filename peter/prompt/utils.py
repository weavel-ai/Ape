from weavel.types import DatasetItem

from peter.types import Dataset
from peter.utils import dict_to_xml


def format_fewshot_xml(fewshot: Dataset) -> str:
    """Format fewshot data into XML format."""
    xml = ""
    for data in fewshot:
        if isinstance(data, DatasetItem):
            inputs, outputs = data.inputs, data.outputs
        else:
            inputs, outputs = data["inputs"], data.get("outputs", {})
        inputs_xml = dict_to_xml(inputs, "inputs")
        outputs_xml = dict_to_xml(outputs, "outputs")
        xml += f"<example>\n{inputs_xml}\n{outputs_xml}\n</example>"

    return xml
