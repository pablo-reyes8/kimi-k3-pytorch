import torch

from conftest import tiny_model
from training import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_reproduces_logits(tmp_path):
    model = tiny_model().eval()
    ids = torch.randint(1, 48, (1, 10))
    expected = model(ids).logits.detach().clone()
    path = save_checkpoint(
        tmp_path / "roundtrip.pt",
        model,
        epoch=2,
        global_step=7,
        model_config=model.config,
    )
    restored = tiny_model().eval()
    state = load_checkpoint(path, restored)
    torch.testing.assert_close(expected, restored(ids).logits)
    assert state["epoch"] == 2 and state["global_step"] == 7
