import pytest

from lor2c.domain.masking import LabelMasker


class TestLabelMasker:
    def test_masks_prompt_prefix(self) -> None:
        assert LabelMasker().mask_prefix(labels=[5, 6, 7, 8], prefix_length=2) == [-100, -100, 7, 8]

    def test_prefix_longer_than_labels_masks_everything(self) -> None:
        assert LabelMasker().mask_prefix(labels=[5, 6], prefix_length=9) == [-100, -100]

    def test_negative_prefix_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            LabelMasker().mask_prefix(labels=[1], prefix_length=-1)

    def test_masks_pad_tokens(self) -> None:
        assert LabelMasker().mask_value(labels=[0, 4, 0, 9], value=0) == [-100, 4, -100, 9]
