"""torch.load with weights_only=True forced: a load that asks for full unpickling raises.

Shared by the exporter shims (yolox.py, dfine.py, rfdetr.py), which call
force_weights_only(torch) before the official exporter loads the checkpoint.
"""


def force_weights_only(torch):
    load = torch.load

    def weights_only_load(*args, **kwargs):
        if kwargs.get("weights_only") is False:
            raise RuntimeError("refusing to unpickle a checkpoint (weights_only=False)")
        return load(*args, **(kwargs | {"weights_only": True}))

    torch.load = weights_only_load
