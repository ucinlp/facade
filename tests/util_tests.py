from util.misc import extract_hypothesis, extract_premise

def test_extract_premise():
    input_1 = ["[CLS]", "I", "love", "food", "[SEP]", "He", "went", "home", ".", "[SEP]"]
    input_2 = ["[CLS]", "I", "[SEP]", "He", "[SEP]"]

    assert extract_premise(input_1) == ["I", "love", "food"]
    assert extract_premise(input_2) == ["I"]

def test_extract_hypothesis():
    input_1 = ["[CLS]", "I", "love", "food", "[SEP]", "He", "went", "home", ".", "[SEP]"]
    input_2 = ["[CLS]", "I", "[SEP]", "He", "[SEP]"]

    assert extract_hypothesis(input_1) == ["He", "went", "home", "."]
    assert extract_hypothesis(input_2) == ["He"]

def main():
    test_extract_hypothesis()
    test_extract_premise()

    print("All Tests Passed.")

if __name__ == "__main__":
    main()