class Plugins:
    def __init__(self, groups):
        self._groups = dict(groups)

        # Expose groups as attributes: plugins.identifiers, plugins.datasets, ...
        for group_name, plugin_map in self._groups.items():
            setattr(self, group_name, plugin_map)

    def keys(self):
        return self._groups.keys()

    def items(self):
        return self._groups.items()

    def __getitem__(self, key):
        return self._groups[key]

    def __str__(self):
        return self.format()

    def asdict(self):
        """
         Return a serializable dict representation of plugins.

         Structure:
        {
             "identifiers": {
                 "uuid4": {
                     "description": "...",
                     "params": [...]
                 },
                 ...
             },
             ...
         }
        """
        result = {}

        for group_name, plugin_map in self._groups.items():
            group_dict = {}
            for plugin_name, plugin in plugin_map.items():
                group_dict[plugin_name] = plugin

            result[group_name] = group_dict

        return result

    def format(self):
        lines = []
        for group_name in sorted(self._groups.keys()):
            plugin_map = self._groups[group_name]
            lines.append(f"{group_name}:")

            if not plugin_map:
                lines.append("  (none)")
                lines.append("")
                continue

            for plugin_name in sorted(plugin_map.keys()):
                plugin = plugin_map[plugin_name]
                description = (getattr(plugin, "description", "") or "").strip()
                params = getattr(plugin, "params", None) or []

                lines.append(f"  - {plugin_name}")
                if description:
                    lines.append(f"      {description}")

                if params:
                    for param in params:
                        param_name = param.get("name")
                        param_type = param.get("type", "any")
                        param_default = param.get("default", None)
                        param_help = (param.get("help", "") or "").strip()
                        param_choices = param.get("choices", None)

                        line = f"      * {param_name} ({param_type})"
                        if param_choices:
                            line += f" choices={param_choices}"
                        if param_default is not None:
                            line += f" default={param_default!r}"
                        if param_help:
                            line += f" — {param_help}"
                        lines.append(line)
                else:
                    lines.append("      (no params)")

                lines.append("")

        return "\n".join(lines).rstrip()
