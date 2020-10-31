# from fvcore.common.registry import Registry

def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    """
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.
    Eg. creating a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_module_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn


from torch import nn

def vgg(num_classes=21,pretrained=True):
    return nn.Sequential(
        nn.Conv2d(3,32,3,1,1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
    )

if __name__=="__main__":
    # from fvcore.common.registry import Registry
    # BACKBONES = Registry("vgg")
    # BACKBONES.register(vgg)
    # print(BACKBONES.get("vgg")())

    BACKBONES = Registry()
    BACKBONES.register("vgg",vgg)
    print(BACKBONES["vgg"]())
