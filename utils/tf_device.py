"""Pick the right TensorFlow device automatically and report it.

CoLight runs on a mix of machines -- some with an NVIDIA GPU, some without. Keras
already places ops on a visible GPU on its own, so there is nothing to *choose* here;
this helper just makes that choice visible (logs CPU vs GPU at startup) and tames GPU
memory use so TF does not grab all VRAM. It never raises -- a probe failure must not
take down a training run that would otherwise be fine on CPU.

Call configure_tf_devices() once, early, before the agent's model is built (so
set_memory_growth still has effect). Logging style mirrors models_inference/LLM/http_llm.py.
"""


def configure_tf_devices():
    """Detect GPUs, enable memory growth, and log which device TF will use."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[Warning] No GPU detected; TensorFlow running on CPU")
            return

        # Memory growth must be set before the GPU is initialized; if TF has already
        # touched it, this raises and we just skip it (TF still uses the GPU).
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

        names = ", ".join(gpu.name for gpu in gpus)
        print(f"[Info] TensorFlow using GPU: {names}")
    except Exception as e:
        # A detection hiccup should never block a run -- fall back to default placement.
        print(f"[Warning] GPU detection failed ({e}); TensorFlow using default device")
