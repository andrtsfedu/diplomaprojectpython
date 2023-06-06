import os
import unittest
from my_program import preprocess_code, tokenize_code, pad_data, create_model, train_model, find_plagiarized_lines

class MyProgramTestCase(unittest.TestCase):
    def setUp(self):
        self.train_dir = 'path/to/train_directory'
        self.test_dir = 'path/to/test_directory'

    def test_preprocess_code(self):
        code = "# This is a comment\nprint('Hello, world!')\n"
        preprocessed_code = preprocess_code(code)
        self.assertEqual(preprocessed_code, "print('Hello, world!')")

    def test_tokenize_code(self):
        code = "print('Hello, world!')"
        tokens = tokenize_code(code)
        self.assertEqual(tokens, ["print", "('Hello,',", "world!')"])

    def test_pad_data(self):
        data = [["print", "('Hello,',", "world!')"]]
        max_sequence_length = 10
        padded_data = pad_data(data, max_sequence_length)
        self.assertEqual(padded_data.shape, (1, 10))

    def test_create_model(self):
        vocab_size = 10000
        embedding_dim = 100
        num_filters = 128
        filter_size = 3
        hidden_dim = 64
        max_sequence_length = 100
        model = create_model(vocab_size, embedding_dim, num_filters, filter_size, hidden_dim, max_sequence_length)
        self.assertIsNotNone(model)

    def test_train_model(self):
        model = train_model(self.train_dir, 10000, 100, 128, 3, 64, 100)
        self.assertIsNotNone(model)

    def test_find_plagiarized_lines(self):
        model = train_model(self.train_dir, 10000, 100, 128, 3, 64, 100)
        self.assertIsNotNone(model)

        file_path = os.path.join(self.test_dir, 'test_file.py')
        plagiarized_lines = find_plagiarized_lines(file_path, model, 10000, 100, 100)
        self.assertIsNotNone(plagiarized_lines)
        self.assertIsInstance(plagiarized_lines, list)

if __name__ == '__main__':
    unittest.main()
