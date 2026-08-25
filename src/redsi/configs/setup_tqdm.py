from tqdm.auto import tqdm


def setup_tqdm():
    _orig_init = tqdm.__init__

    l_bar = "{desc:40.40}: {percentage:3.0f}%|"
    bar = "{bar:100}"
    r_bar = "| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    bar_format = f"{l_bar}{bar}{r_bar}"

    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("bar_format", bar_format)
        return _orig_init(self, *args, **kwargs)

    tqdm.__init__ = _patched_init
