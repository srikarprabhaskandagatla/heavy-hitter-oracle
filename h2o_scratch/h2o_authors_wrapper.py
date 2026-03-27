import sys
from pathlib import Path

_AUTHORS_H2O_PATH = Path(__file__).parent.parent / "h2o_authors" / "h2o_hf"

if str(_AUTHORS_H2O_PATH) not in sys.path:
    sys.path.insert(0, str(_AUTHORS_H2O_PATH))

try:
    from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent
except ImportError as e:
    raise ImportError(
        "\n[h2o_authors_wrapper] Could not import the authors' code.\n"
        "Make sure you have run:\n"
        "    git clone https://github.com/FMInference/H2O.git h2o_authors\n"
        "from the project root directory.\n"
        f"Original error: {e}"
    )


def patch_model_with_authors_h2o(model,
                                  heavy_ratio: float = 0.10,
                                  recent_ratio: float = 0.10):
    """
    Apply the authors' KV-eviction patch to an OPT model in-place.
    """
    model.config.heavy_ratio  = heavy_ratio
    model.config.recent_ratio = recent_ratio

    model = convert_kvcache_opt_heavy_recent(model, model.config)

    model = model.half().cuda()

    print(f"[Authors' H2O] Patched model "
          f"(heavy={heavy_ratio:.0%}, recent={recent_ratio:.0%}, "
          f"total budget={heavy_ratio + recent_ratio:.0%})")
    return model