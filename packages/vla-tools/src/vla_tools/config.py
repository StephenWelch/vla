import sys


def configuration_arguments(values, prefix=""):
    result = []
    for key, value in values.items():
        name = prefix + key.replace("_", "-")
        if isinstance(value, dict):
            result.extend(configuration_arguments(value, name + "."))
        elif isinstance(value, bool):
            result.append(
                "--" + (name if value else prefix + "no-" + key.replace("_", "-"))
            )
        else:
            result.append("--" + name)
            result.extend(
                "None" if item is None else str(item)
                for item in (value if isinstance(value, list) else [value])
            )
    return result


def parse_args(config_type, argv=None):
    """Typed defaults < YAML < explicit CLI flags."""
    import tyro
    from omegaconf import OmegaConf

    argv = list(sys.argv[1:] if argv is None else argv)
    values = OmegaConf.create({})
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--config" or token.startswith("--config="):
            if token == "--config":
                if index + 1 == len(argv):
                    raise ValueError("--config requires a YAML path")
                path = argv.pop(index + 1)
            else:
                path = token.split("=", 1)[1]
            argv.pop(index)
            loaded = OmegaConf.load(path)
            if not OmegaConf.is_dict(loaded):
                raise ValueError("Configuration must be a YAML mapping")
            values = OmegaConf.merge(values, loaded)
        else:
            index += 1
    argv = configuration_arguments(OmegaConf.to_container(values, resolve=True)) + argv
    return tyro.cli(config_type, args=argv, description=config_type.__doc__)
