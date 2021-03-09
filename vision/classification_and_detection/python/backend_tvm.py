"""
TVM backend (https://github.com/apache/tvm)
"""

# tvm, relay
import tvm
from tvm import te
from tvm import relay

from tvm.contrib import graph_runtime

# Tensorflow imports
import tensorflow as tf
import backend

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

class BackendTvm(backend.Backend):
    def __init__(self, target="llvm", target_host="llvm", layout=None, context=tvm.cpu(0)):
        super(BackendTvm, self).__init__()
        self.target = target
        self.target_host = target_host
        self.layout = layout
        self.ctx = context

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tvm"

    def image_format(self):
        # tflite is always NHWC
        return "None"

    def load(self, model_path, inputs=None, outputs=None):
        # there is no input/output meta data i the graph so it need to come from config.
        if not inputs:
            raise ValueError("TVM needs inputs")
        if not outputs:
            raise ValueError("TVM needs outputs")
        self.outputs = outputs
        self.inputs = inputs

        # Import model
        # ------------
        # Creates tensorflow graph definition from protobuf file.

        with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            g = tf_compat_v1.import_graph_def(graph_def, name="")
        self.sess = tf_compat_v1.Session(graph=g)

        # Import the graph to Relay
        # -------------------------
        # Import tensorflow graph definition to relay frontend.
        shape_dict = {"DecodeJpeg/contents": x.shape}
        mod, params = relay.frontend.from_tensorflow(graph_def, layout=self.layout, shape=shape_dict)

        # Relay Build
        # -----------
        # Compile the graph to llvm target with given input specification.
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=self.target, target_host=self.target_host, params=params)

        self.m = graph_runtime.GraphModule(lib["default"](self.ctx))

        return self

    def predict(self, feed):
        return self.m.run()
