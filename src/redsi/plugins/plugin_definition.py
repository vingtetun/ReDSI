from dataclasses import dataclass
from typing import Any

# Errors
ERR_PLUGIN_ATTRIBUTE_MISSING = "[{identifier}] has missing '{attr}' attribute."
ERR_PLUGIN_ATTRIBUTE_TYPE = (
    "[{identifier}] '{attr}' attribute type unexpected.Expected '{expected_type}', got '{value_type}'."
)


@dataclass(frozen=True)
class PluginAttribute:
    name: str
    expected_type: Any


REQUIRED_PLUGIN_ATTRIBUTES = [
    PluginAttribute("description", expected_type=str),
    PluginAttribute("params", expected_type=list),
    PluginAttribute("run", expected_type=callable),
]


def check_plugin(identifier, plugin):
    for attribute in REQUIRED_PLUGIN_ATTRIBUTES:
        check_plugin_attribute(identifier, plugin, attribute)


def check_plugin_attribute(identifier, plugin, attribute):
    name = attribute.name

    if not hasattr(plugin, name):
        msg = ERR_PLUGIN_ATTRIBUTE_MISSING.format(identifier=identifier, attr=name)
        raise ValueError(msg)

    value = getattr(plugin, name)

    msg = None
    if attribute.expected_type is callable:
        if not callable(value):
            msg = ERR_PLUGIN_ATTRIBUTE_TYPE.format(
                identifier=identifier, attr=name, expected_type="callable", value_type=type(value)
            )
    else:
        if not isinstance(value, attribute.expected_type):
            msg = ERR_PLUGIN_ATTRIBUTE_TYPE.format(
                identifier=identifier, attr=name, expected_type=attribute.expected_type, value_type=type(value)
            )

    if msg is not None:
        raise TypeError(msg)
