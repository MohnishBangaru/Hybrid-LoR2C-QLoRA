from lor2c.domain.caption import CaptionNormalizer


class TestCaptionNormalizer:
    def test_strips_numbered_prefixes(self) -> None:
        normalizer = CaptionNormalizer()
        assert normalizer.normalize(text="12. A cat on a mat") == "A cat on a mat"
        assert normalizer.normalize(text="3: girl with red hair") == "girl with red hair"
        assert normalizer.normalize(text="  7 - anime scene ") == "anime scene"

    def test_leaves_plain_captions_alone(self) -> None:
        assert CaptionNormalizer().normalize(text="Two dogs running") == "Two dogs running"
