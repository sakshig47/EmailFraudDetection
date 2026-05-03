import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from src.data.preprocess import (
    to_lowercase, remove_html_tags, remove_urls, remove_emails,
    remove_phone_numbers, remove_special_characters,
    tokenize, remove_stopwords, stem_tokens, preprocess_text,
)

class TestToLowercase:
    def test_basic(self): assert to_lowercase("HELLO WORLD") == "hello world"
    def test_mixed(self): assert to_lowercase("HeLLo") == "hello"
    def test_empty(self): assert to_lowercase("") == ""

class TestRemoveHtmlTags:
    def test_bold(self):
        r = remove_html_tags("<b>hello</b>")
        assert "<b>" not in r and "hello" in r
    def test_no_tags(self): assert remove_html_tags("plain") == "plain"

class TestRemoveUrls:
    def test_http(self):
        r = remove_urls("Visit http://example.com now")
        assert "http://example.com" not in r and "url" in r
    def test_www(self):
        r = remove_urls("Check www.spam.biz")
        assert "www.spam.biz" not in r
    def test_no_url(self): assert remove_urls("no url here") == "no url here"

class TestRemoveEmails:
    def test_basic(self):
        r = remove_emails("contact spam@x.com")
        assert "@" not in r and "email" in r
    def test_no_email(self): assert remove_emails("no email") == "no email"

class TestRemoveSpecialCharacters:
    def test_punctuation(self):
        r = remove_special_characters("hello!!! world???")
        assert "!" not in r and "?" not in r
    def test_keeps_spaces(self): assert " " in remove_special_characters("hello, world!")

class TestTokenize:
    def test_basic(self): assert tokenize("hello world") == ["hello", "world"]
    def test_empty(self): assert tokenize("") == []

class TestRemoveStopwords:
    def test_removes_the(self):
        r = remove_stopwords(["the", "quick", "fox"])
        assert "the" not in r and "quick" in r
    def test_empty(self): assert remove_stopwords([]) == []

class TestStemTokens:
    def test_returns_list(self):
        r = stem_tokens(["running", "prizes"])
        assert isinstance(r, list) and len(r) == 2
    def test_empty(self): assert stem_tokens([]) == []

class TestPreprocessText:
    def test_spam(self):
        r = preprocess_text("WINNER!! FREE PRIZE Call 09061701461 www.win.biz")
        assert isinstance(r, str) and len(r) > 0
        assert "http" not in r
    def test_html(self):
        r = preprocess_text("<b>Click</b> to verify your account now!")
        assert "<b>" not in r
    def test_empty(self): assert preprocess_text("") == ""
    def test_lowercase(self):
        r = preprocess_text("HELLO FREE MONEY")
        assert r == r.lower()
    def test_no_stopwords(self):
        r = preprocess_text("the quick brown fox jumps over the lazy dog")
        assert "the" not in r.split()
    def test_url_removed(self):
        r = preprocess_text("visit https://phishing.biz/login now")
        assert "https" not in r
    def test_returns_string(self):
        for t in ["hello", "SPAM", "", "123"]:
            assert isinstance(preprocess_text(t), str)

class TestPrediction:
    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        from src.config import MODEL_PATH
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Run `python main.py` first to train the model.")

    def test_spam_detected(self):
        from src.models.predict import predict_single
        r = predict_single("WINNER FREE PRIZE click http://win.biz call now!")
        assert r["spam_prob"] > 0.5

    def test_ham_detected(self):
        from src.models.predict import predict_single
        r = predict_single("Hi, are you free for a meeting tomorrow afternoon?")
        assert r["spam_prob"] < 0.5

    def test_output_keys(self):
        from src.models.predict import predict_single
        r = predict_single("test email")
        for k in ["label","label_index","confidence","spam_prob"]: assert k in r

    def test_probabilities_range(self):
        from src.models.predict import predict_single
        r = predict_single("free money prize win click")
        assert 0.0 <= r["spam_prob"] <= 1.0
        assert 0.0 <= r["confidence"] <= 1.0

    def test_label_valid(self):
        from src.models.predict import predict_single
        from src.config import LABEL_NAMES
        r = predict_single("Hello how are you doing today?")
        assert r["label"] in LABEL_NAMES

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
