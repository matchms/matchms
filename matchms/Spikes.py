from deprecated.sphinx import deprecated
from matchms.Fragments import Fragments


@deprecated(version='0.14.0', reason="Class was remaned to `Fragments`.")
class Spikes(Fragments):
    pass
