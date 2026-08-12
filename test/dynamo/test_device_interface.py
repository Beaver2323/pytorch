# Owner(s): ["module: dynamo"]

import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import device_interface
from torch._dynamo.device_interface import (
    CpuInterface,
    CudaInterface,
    DeviceInterface,
    get_device_autocast_classes,
    get_registered_device_interfaces,
    MpsInterface,
    MtiaInterface,
    register_interface_for_device,
    XpuInterface,
)
from torch._dynamo.variables.torch import (
    _matches_device_autocast_class,
    device_type_for_autocast_class,
)
from torch._dynamo.variables.user_defined import UserDefinedClassVariable


# The stub backend registers under a real device_type, because torch.amp
# rejects unknown ones, and asserting on the traced device_type is the point of
# the autocast tests below.
STUB_DEVICE = "privateuseone"


class _StubStream(torch.Stream):
    """A well-behaved backend stream: subclasses torch.Stream."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)


class _StubEvent(torch.Event):
    """A well-behaved backend event: subclasses torch.Event."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)


class _StubTensorType:
    """Stand-in for a backend tensor constructor such as torch.foo.FloatTensor."""


class _StubAutocast(torch.amp.autocast_mode.autocast):
    """A backend's autocast, whose device_type is implicit in the class itself.

    Note it is passed to super() but never declared to the interface: the
    device_type Dynamo traces has to come from the registration key.
    """

    def __init__(self, dtype=torch.float16, enabled=True, cache_enabled=None):
        super().__init__(
            STUB_DEVICE, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
        )


class GoodStubInterface(DeviceInterface):
    Stream = _StubStream
    Event = _StubEvent
    tensor_types = frozenset({_StubTensorType})
    autocast_classes = frozenset({_StubAutocast})


class BadStubInterface(DeviceInterface):
    """A backend that forgot to subclass torch.Stream/torch.Event.

    It inherits DeviceInterface's placeholders, which only raise
    NotImplementedError, so they must never reach _in_graph_classes().
    """


class DeviceInterfaceSlotsTestCase(torch._dynamo.test_case.TestCase):
    """Shared registration/teardown for the two declarative slots."""

    def tearDown(self):
        for name in (STUB_DEVICE, f"{STUB_DEVICE}:3", "badstub"):
            device_interface.device_interfaces.pop(name, None)
        UserDefinedClassVariable._in_graph_classes.cache_clear()
        get_device_autocast_classes.cache_clear()
        super().tearDown()

    def _register(self, name, iface):
        register_interface_for_device(name, iface)
        self.assertIs(dict(get_registered_device_interfaces())[name], iface)


class InGraphClassesTests(DeviceInterfaceSlotsTestCase):
    def test_registered_interface_stream_event_in_graph(self):
        # Prime the cache first: this is the out-of-tree backend's situation,
        # where registration happens lazily, after Dynamo has already run.
        before = UserDefinedClassVariable._in_graph_classes()
        self.assertNotIn(_StubStream, before)
        self.assertNotIn(_StubEvent, before)

        self._register(STUB_DEVICE, GoodStubInterface)

        after = UserDefinedClassVariable._in_graph_classes()
        self.assertIn(_StubStream, after)
        self.assertIn(_StubEvent, after)

    def test_registered_interface_tensor_types_in_graph(self):
        self.assertNotIn(_StubTensorType, UserDefinedClassVariable._in_graph_classes())

        self._register(STUB_DEVICE, GoodStubInterface)

        self.assertIn(_StubTensorType, UserDefinedClassVariable._in_graph_classes())

    def test_base_class_placeholders_never_in_graph(self):
        self._register("badstub", BadStubInterface)

        in_graph = UserDefinedClassVariable._in_graph_classes()
        # The placeholders raise NotImplementedError with a message telling the
        # backend to subclass torch.Stream/torch.Event.  Putting them in this
        # set would trade that message for an opaque tracing failure.
        self.assertNotIn(DeviceInterface.Stream, in_graph)
        self.assertNotIn(DeviceInterface.Event, in_graph)
        self.assertNotIn(BadStubInterface.Stream, in_graph)
        self.assertNotIn(BadStubInterface.Event, in_graph)

        for cls in in_graph:
            self.assertIsInstance(cls, type)

    def test_slots_default_to_empty(self):
        # The base-class defaults must be inert, so declaring them changes
        # nothing for backends that do not opt in.  Checked against the in-tree
        # interfaces specifically: an out-of-tree backend may well be
        # registered, and is entitled to declare either slot.
        self.assertEqual(DeviceInterface.tensor_types, frozenset())
        self.assertEqual(DeviceInterface.autocast_classes, frozenset())
        for iface in (
            CudaInterface,
            XpuInterface,
            MtiaInterface,
            CpuInterface,
            MpsInterface,
        ):
            self.assertEqual(iface.tensor_types, frozenset())
            self.assertEqual(iface.autocast_classes, frozenset())
            self.assertNotIn("tensor_types", vars(iface))
            self.assertNotIn("autocast_classes", vars(iface))

    def test_late_registration_invalidates_cache(self):
        # _in_graph_classes() is functools.cache'd over a mutable registry, so
        # register_interface_for_device() has to invalidate it.
        UserDefinedClassVariable._in_graph_classes()
        self._register(STUB_DEVICE, GoodStubInterface)
        self.assertIn(_StubStream, UserDefinedClassVariable._in_graph_classes())

    def test_stream_construction_traced_after_registration(self):
        self._register(STUB_DEVICE, GoodStubInterface)

        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x):
            _StubStream()
            return x + 1

        fn(torch.ones(2))
        self.assertEqual(counter.frame_count, 1)

    @unittest.skipIf(not torch.cuda._is_compiled(), "CUDA not compiled in")
    def test_cuda_stream_event_still_in_graph(self):
        # torch.cuda.Stream/Event used to be listed here by hand.  CudaInterface
        # supplies them now, so the set must be unchanged for a CUDA build.
        # _CudaStreamBase's tp_base is torch.Stream, so the issubclass() check
        # in the loop admits them.
        in_graph = UserDefinedClassVariable._in_graph_classes()
        self.assertIn(torch.cuda.Stream, in_graph)
        self.assertIn(torch.cuda.Event, in_graph)

    @unittest.skipIf(not torch.xpu._is_compiled(), "XPU not compiled in")
    def test_xpu_stream_event_still_in_graph(self):
        in_graph = UserDefinedClassVariable._in_graph_classes()
        self.assertIn(torch.xpu.Stream, in_graph)
        self.assertIn(torch.xpu.Event, in_graph)


class DeviceAutocastTests(DeviceInterfaceSlotsTestCase):
    def test_matches_only_when_registered(self):
        # Matching is by declaration, not by issubclass: an arbitrary autocast
        # subclass must not be picked up unless its owning device registered it.
        self.assertFalse(_matches_device_autocast_class(_StubAutocast))

        self._register(STUB_DEVICE, GoodStubInterface)

        self.assertTrue(_matches_device_autocast_class(_StubAutocast))

    def test_device_type_derived_from_registration_key(self):
        # GoodStubInterface declares which class is its autocast entry point,
        # never which device_type it means, so this can only come from the key.
        self.assertIsNone(device_type_for_autocast_class(_StubAutocast))

        self._register(STUB_DEVICE, GoodStubInterface)

        self.assertEqual(device_type_for_autocast_class(_StubAutocast), STUB_DEVICE)
        self.assertEqual(get_device_autocast_classes()[_StubAutocast], STUB_DEVICE)

    def test_device_type_strips_index_from_key(self):
        # Backends register one interface per device index ("npu", "npu:0", ...),
        # and an index is not part of a device_type.
        self._register(f"{STUB_DEVICE}:3", GoodStubInterface)

        self.assertEqual(device_type_for_autocast_class(_StubAutocast), STUB_DEVICE)

    def test_base_autocast_never_matches(self):
        # The generic base takes device_type as an argument, so it belongs to no
        # device.  It is also in supported_ctx_manager_classes already, and must
        # keep reaching that branch rather than this one.
        self._register(STUB_DEVICE, GoodStubInterface)

        self.assertFalse(
            _matches_device_autocast_class(torch.amp.autocast_mode.autocast)
        )

    def test_internal_autocast_subclass_not_matched(self):
        # _UnmanagedAutocast is the class _enter_autocast() constructs during
        # pre-dispatch tracing.  It subclasses autocast but belongs to no
        # device, so a too-broad match would route it to AutocastModeVariable
        # and blow up test__enter__exit_autocast with "setattr() on
        # unsupported type".
        from torch.amp.autocast_mode import _UnmanagedAutocast

        self._register(STUB_DEVICE, GoodStubInterface)

        self.assertFalse(_matches_device_autocast_class(_UnmanagedAutocast))

    def test_late_registration_invalidates_autocast_cache(self):
        # get_device_autocast_classes() is functools.cache'd over the same
        # mutable registry, so it needs the same invalidation.
        get_device_autocast_classes()
        self._register(STUB_DEVICE, GoodStubInterface)
        self.assertIn(_StubAutocast, get_device_autocast_classes())

    def test_registered_autocast_class_traced(self):
        self._register(STUB_DEVICE, GoodStubInterface)

        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x):
            with _StubAutocast():
                return x + 1

        fn(torch.ones(2))
        self.assertEqual(counter.frame_count, 1)

    def test_traced_graph_carries_derived_device_type(self):
        # The end-to-end statement of the above: the device_type that reaches
        # the graph is the registration key, with no string in the class.
        self._register(STUB_DEVICE, GoodStubInterface)

        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            with _StubAutocast():
                return x + 1

        fn(torch.ones(2))
        self.assertEqual(len(graphs), 1)
        enters = [
            node
            for node in graphs[0].graph.nodes
            if node.op == "call_function" and node.target is torch.amp._enter_autocast
        ]
        self.assertEqual(len(enters), 1)
        self.assertEqual(enters[0].args[0], STUB_DEVICE)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
