import os
import sys
import types
import pytest

# Provide dummy PIL module if Pillow is not installed
if 'PIL' not in sys.modules:
    dummy = types.ModuleType('PIL')
    class DummyImage:
        pass
    DummyImage.Image = DummyImage  # allow Image.Image references
    dummy.Image = DummyImage
    sys.modules['PIL'] = dummy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.prompt_utils import normalize_emotion

@pytest.mark.parametrize(
    "inp,expected",
    [
        ("سعاده", "سعادة"),
        ("فرح", "سعادة"),
        ("ثقه", "ثقة"),
        ("رعب", "خوف"),
        ("مفاجاة", "مفاجأة"),
        ("كآبة", "حزن"),
        ("اشمئزاز", "قرف"),
        ("سخط", "غضب"),
        ("انتظار", "ترقب"),
        ("حيادي", "محايد"),
        ("سُعَادَةٌ!!", "سعادة"),
        ("إستياء", "قرف"),
    ]
)
def test_normalize_emotion_variations(inp, expected):
    assert normalize_emotion(inp) == expected
